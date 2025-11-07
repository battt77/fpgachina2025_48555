"""
RMSNorm Acceleration Circuit Evaluation
PCIe + UDP test: Send data via PCIe, trigger FPGA via UDP, verify results
"""

import sys
import numpy as np
import time
import torch
import RMS_BP_tch as rms
import socket
import struct

# =============================================================================
# Configuration
# =============================================================================
X_DIM = 4        # sequence length
Y_DIM = 512      # hidden states
BATCH_SIZE = 8   # transfer batch size
BLOCK_SIZE = int(Y_DIM/BATCH_SIZE)

# Accuracy tolerance
RTOL = 1e-2      # relative tolerance
ATOL = 1e-2      # absolute tolerance

# PCIe functions
from pcie_rw import (
    check_xdma_driver, pcie_init, pcie_open, pcie_close,
    write_fp16_array_to_ddr, c2h_transfer,
    ADDR_X_DATA, ADDR_WEIGHTS, ADDR_BIAS, ADDR_GRAD
)

# DDR output addresses
ADDR_OUTPUT = 0x00500000       # forward output (5MB offset)
ADDR_GRAD_X = 0x00600000       # input gradient (6MB offset)
ADDR_GRAD_WEIGHT = 0x00700000  # weight gradient (7MB offset)
ADDR_GRAD_BIAS = 0x00800000    # bias gradient (8MB offset)
ADDR_CAL_CLK = 0x00900000      # clock cycles (9MB offset)

# UDP client
from udp_client import UDPClient

# UDP configuration
BOARD_IP = "192.168.1.10"
BOARD_PORT = 5001
LOCAL_IP = "192.168.1.20"
LOCAL_PORT = 8080
COMPLETION_SIGNAL = 0x0001

# FPGA clock frequency
FPGA_CLK_FREQ_MHZ = 250.0  # 250MHz


class My_net(torch.nn.Module):
  def __init__(self):
    super(My_net,self).__init__()
    self.myrms=rms.RMSNormTrue(Y_DIM,eps=1e-6,batch_size=BLOCK_SIZE)
  def forward(self,x):
    y = self.myrms(x)
    return y
  
def send_all_data_via_pcie():
    """
    Send data to DDR via PCIe and compute CPU reference
    Returns: (success, ln_y, delta_x, delta_weight, delta_bias, fw_time, bw_time)
    """
    print(f"\nStep 1: Send data to DDR via PCIe")
    
    try:
        # 1. Input data
        print(f"[1/4] Transferring input data...")
        np.random.seed(42)
        x_data = np.random.randn(X_DIM * Y_DIM).astype(np.float16)
        success, total = write_fp16_array_to_ddr(x_data, ADDR_X_DATA, BATCH_SIZE)
        print(f"  ✓ Input data transferred: {success}/{total} batches")
        if success != total:
            return (False, None, None, None, None, 0, 0)
        
        # 2. Weights
        print(f"[2/4] Transferring weights...")
        np.random.seed(123)
        weights_single = np.random.uniform(0.8, 1.2, Y_DIM).astype(np.float16)
        weights = np.tile(weights_single, X_DIM)
        success, total = write_fp16_array_to_ddr(weights, ADDR_WEIGHTS, BATCH_SIZE)
        print(f"  ✓ Weights transferred: {success}/{total} batches")
        if success != total:
            return (False, None, None, None, None, 0, 0)
        
        # 3. Bias
        print(f"[3/4] Transferring bias...")
        bias_single = np.zeros(Y_DIM, dtype=np.float16)
        bias = np.tile(bias_single, X_DIM)
        success, total = write_fp16_array_to_ddr(bias, ADDR_BIAS, BATCH_SIZE)
        print(f"  ✓ Bias transferred: {success}/{total} batches")
        if success != total:
            return (False, None, None, None, None, 0, 0)
        
        # Compute CPU reference
        print(f"\n[CPU Reference] PyTorch computing...")
        device = torch.device('cpu')
        net = My_net().to(device).half()
        weight_tensor = torch.from_numpy(weights_single).to(device)
        x_tensor = torch.from_numpy(x_data.reshape(X_DIM, Y_DIM)).to(device).requires_grad_(True)
        params_init_dict = {'myrms.weight': weight_tensor}
        net.load_state_dict(params_init_dict)
        
        fw_start = time.perf_counter()
        ln_y = net(x_tensor)
        fw_end = time.perf_counter()
        fw_time = (fw_end - fw_start) * 1000

        log_probs = torch.nn.functional.log_softmax(ln_y, dim=-1)
        per_example_loss = -torch.sum(log_probs, dim=-1)
        loss = torch.mean(per_example_loss)
        
        bw_start = time.perf_counter()
        loss.backward()
        bw_end = time.perf_counter()
        bw_time = (bw_end - bw_start) * 1000
        
        delta_x = x_tensor.grad
        verify_ln_y_grad_value = torch.tensor(ln_y.detach(), requires_grad=True, dtype=torch.float16)
        log_probs3 = torch.nn.functional.log_softmax(verify_ln_y_grad_value, dim=-1)
        per_example_loss3 = -torch.sum(log_probs3, dim=-1)
        loss_3 = torch.mean(per_example_loss3)
        loss_3.backward()
        delta_ln_y = verify_ln_y_grad_value.grad
        
        delta_weight = None
        delta_bias = None
        for name, parms in net.named_parameters():
            if name == "myrms.weight":
                delta_weight = parms.grad

        # 4. Output gradient
        print(f"[4/4] Transferring output gradient...")
        grad = delta_ln_y.detach().cpu().numpy().astype(np.float16).flatten()
        success, total = write_fp16_array_to_ddr(grad, ADDR_GRAD, BATCH_SIZE)
        print(f"  ✓ Gradient transferred: {success}/{total} batches")
        if success != total:
            return (False, None, None, None, None, 0, 0)
        
        print(f"✓ PCIe transfer & CPU compute done (FW: {fw_time:.3f}ms, BW: {bw_time:.3f}ms)")
        
        return (True, ln_y, delta_x, delta_weight, delta_bias, fw_time, bw_time)
        
    except Exception as e:
        print(f"\n❌ PCIe transfer error: {e}")
        return (False, None, None, None, None, 0, 0)


def read_and_compare_results(reference_output, ref_delta_x, ref_delta_weight, ref_delta_bias, cpu_fw_time, cpu_bw_time, iteration=None, total_iterations=None):
    """
    Read FPGA results and compare with CPU reference
    Returns: (fpga_fw_time, fpga_bw_time, accuracy_dict) or None on failure
    """
    if iteration and total_iterations:
        print(f"\nStep 3: Read FPGA results and verify (Iteration {iteration}/{total_iterations})")
    else:
        print(f"\nStep 3: Read FPGA results and verify")
    
    try:
        # ====== 1. Read forward output ======
        print(f"\n[1/4] Reading forward output...")
        output_size = X_DIM * Y_DIM * 2
        fpga_output_bytes = c2h_transfer(ADDR_OUTPUT, output_size)
        
        if fpga_output_bytes is None:
            print("❌ Failed to read from FPGA")
            return False
        
        fpga_output_fp16 = np.frombuffer(fpga_output_bytes, dtype=np.float16)
        fpga_output_fp16 = fpga_output_fp16.reshape(X_DIM, Y_DIM)
        fpga_output_tensor = torch.from_numpy(fpga_output_fp16)
        
        fw_is_close = torch.allclose(reference_output.cpu().detach(), fpga_output_tensor, rtol=RTOL, atol=ATOL)
        diff_fw = torch.abs(reference_output.cpu().detach() - fpga_output_tensor)
        
        if fw_is_close:
            print(f"  ✓ Forward output verified (rtol={RTOL}, atol={ATOL})")
        else:
            print(f"  ❌ Forward output mismatch (max_diff: {torch.max(diff_fw).item():.6f})")
        
        # ====== 2. Read input gradient ======
        print(f"[2/4] Reading input gradient...")
        grad_x_size = X_DIM * Y_DIM * 2
        fpga_grad_x_bytes = c2h_transfer(ADDR_GRAD_X, grad_x_size)
        
        if fpga_grad_x_bytes is None:
            print("❌ Failed to read input gradient")
            return False
        
        fpga_grad_x = np.frombuffer(fpga_grad_x_bytes, dtype=np.float16)
        fpga_grad_x = fpga_grad_x.reshape(X_DIM, Y_DIM)
        fpga_grad_x_tensor = torch.from_numpy(fpga_grad_x)
        
        grad_x_is_close = torch.allclose(ref_delta_x.cpu().detach(), fpga_grad_x_tensor, rtol=RTOL, atol=ATOL)
        diff_grad_x = torch.abs(ref_delta_x.cpu().detach() - fpga_grad_x_tensor)
        
        if grad_x_is_close:
            print(f"  ✓ Input gradient verified (rtol={RTOL}, atol={ATOL})")
        else:
            print(f"  ❌ Input gradient mismatch (max_diff: {torch.max(diff_grad_x).item():.6f})")
        
        # ====== 3. Read weight gradient ======
        print(f"[3/4] Reading weight gradient...")
        grad_weight_size = Y_DIM * 2
        fpga_grad_weight_bytes = c2h_transfer(ADDR_GRAD_WEIGHT, grad_weight_size)
        
        if fpga_grad_weight_bytes is None:
            print("❌ Failed to read weight gradient")
            return False
        
        fpga_grad_weight = np.frombuffer(fpga_grad_weight_bytes, dtype=np.float16)
        fpga_grad_weight_tensor = torch.from_numpy(fpga_grad_weight)
        
        weight_is_close = torch.allclose(ref_delta_weight.cpu().detach(), fpga_grad_weight_tensor, rtol=RTOL, atol=ATOL)
        diff_weight = torch.abs(ref_delta_weight.cpu().detach() - fpga_grad_weight_tensor)
        
        if weight_is_close:
            print(f"  ✓ Weight gradient verified (rtol={RTOL}, atol={ATOL})")
        else:
            print(f"  ❌ Weight gradient mismatch (max_diff: {torch.max(diff_weight).item():.6f})")
        
        # ====== 4. Bias gradient (skip for RMS) ======
        print(f"[4/4] Bias gradient (skip, RMS has no bias)")
        bias_is_close = True
        
        # ====== 5. Read FPGA timing ======
        clk_size = 12
        fpga_clk_bytes = c2h_transfer(ADDR_CAL_CLK, clk_size)
        
        fpga_fw_time_ms = 0
        fpga_bw_time_ms = 0
        
        if fpga_clk_bytes is not None and len(fpga_clk_bytes) == clk_size:
            fm_cycles = struct.unpack('<I', fpga_clk_bytes[0:4])[0]
            bm_param_cycles = struct.unpack('<I', fpga_clk_bytes[4:8])[0]  # weight gradient backprop
            bm_cycles = struct.unpack('<I', fpga_clk_bytes[8:12])[0]        # input gradient backprop
            
            fm_time_ms = (fm_cycles / (FPGA_CLK_FREQ_MHZ * 1e6)) * 1000.0
            bm_param_time_ms = (bm_param_cycles / (FPGA_CLK_FREQ_MHZ * 1e6)) * 1000.0
            bm_time_ms = (bm_cycles / (FPGA_CLK_FREQ_MHZ * 1e6)) * 1000.0
            fpga_bw_time_ms = bm_param_time_ms + bm_time_ms  # Total backward = param grad + input grad
            fpga_fw_time_ms = fm_time_ms
            fpga_total_time = fm_time_ms + fpga_bw_time_ms
            cpu_total_time = cpu_fw_time + cpu_bw_time
            
            # Performance & Accuracy Report (only show if not in multi-iteration mode)
            if not iteration:
                print(f"\n{'='*70}")
                print(f"{'RMS Acceleration Circuit Evaluation':^70}")
                print(f"{'='*70}")
                
                # Accuracy comparison (using rtol and atol thresholds)
                print(f"\n[Accuracy Comparison] (Threshold: rtol={RTOL}, atol={ATOL})")
                print(f"  Forward Output    : {'PASS' if fw_is_close else 'FAIL':>8}  (max_diff: {torch.max(diff_fw).item():.6f})")
                print(f"  Input Gradient    : {'PASS' if grad_x_is_close else 'FAIL':>8}  (max_diff: {torch.max(diff_grad_x).item():.6f})")
                print(f"  Weight Gradient   : {'PASS' if weight_is_close else 'FAIL':>8}  (max_diff: {torch.max(diff_weight).item():.6f})")
                
                # Performance comparison
                print(f"\n[Performance Comparison]")
                print(f"{'':20} {'CPU (ms)':>12} {'FPGA (ms)':>12} {'Speedup':>12}")
                print(f"{'-'*70}")
                
                fw_speedup = cpu_fw_time / fm_time_ms if fm_time_ms > 0 else 0
                bw_speedup = cpu_bw_time / fpga_bw_time_ms if fpga_bw_time_ms > 0 else 0
                total_speedup = cpu_total_time / fpga_total_time if fpga_total_time > 0 else 0
                
                print(f"  {'Forward':<18} {cpu_fw_time:>12.3f} {fm_time_ms:>12.3f} {fw_speedup:>11.2f}x")
                print(f"  {'Backward':<18} {cpu_bw_time:>12.3f} {fpga_bw_time_ms:>12.3f} {bw_speedup:>11.2f}x")
                print(f"{'-'*70}")
                print(f"  {'Total':<18} {cpu_total_time:>12.3f} {fpga_total_time:>12.3f} {total_speedup:>11.2f}x")
                
                print(f"{'='*70}")
                
                # Note: FPGA Backward includes both weight gradient and input gradient backprop
                print(f"\nNote: FPGA Backward = Weight Grad ({bm_param_time_ms:.3f}ms) + Input Grad ({bm_time_ms:.3f}ms)")
            else:
                # In multi-iteration mode, just show brief status
                print(f"  ✓ Timing recorded (FW: {fm_time_ms:.5f}ms, BW: {fpga_bw_time_ms:.5f}ms)")
        
        # Summary
        all_match = fw_is_close and grad_x_is_close and weight_is_close
        
        if all_match:
            if not iteration:
                print(f"\n✓ All verification passed!")
        else:
            if not iteration:
                print(f"\n❌ Verification failed!")
        
        # Return timing and accuracy data
        accuracy_dict = {
            'fw': fw_is_close,
            'grad_x': grad_x_is_close,
            'weight': weight_is_close
        }
        return (fpga_fw_time_ms, fpga_bw_time_ms, accuracy_dict)
            
    except Exception as e:
        print(f"\n❌ Read/compare error: {e}")
        import traceback
        traceback.print_exc()
        return None


def send_activate_via_udp():
    """
    Send activation command via UDP and wait for completion
    Returns: True=success, False=fail
    """
    print(f"\nStep 2: Send activation via UDP and wait")
    
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((LOCAL_IP, LOCAL_PORT))
        
        activate_cmd = struct.pack('<I', 0x00000001)
        sock.sendto(activate_cmd, (BOARD_IP, BOARD_PORT))
        print(f"✓ Activation sent, waiting for FPGA...")
        
        start_time = time.time()
        while True:
            data, addr = sock.recvfrom(1024)
            
            if len(data) >= 2:
                received_value = struct.unpack('<H', data[:2])[0]
                
                if received_value == COMPLETION_SIGNAL:
                    elapsed = time.time() - start_time
                    print(f"✓ FPGA processing done (elapsed: {elapsed:.3f}s)")
                    return True
        
    except KeyboardInterrupt:
        print("\n\nUser interrupted")
        return False
    except Exception as e:
        print(f"\n❌ UDP error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if sock:
            sock.close()


def main():
    NUM_ITERATIONS = 10  # Number of test iterations
    
    print("\n" + "="*70)
    print("RMSNorm FPGA Acceleration Test (Config: {}x{}, Batch={})".format(X_DIM, Y_DIM, BATCH_SIZE))
    print(f"Running {NUM_ITERATIONS} iterations for average performance...")
    print("="*70)
    
    # Initialize PCIe
    check_xdma_driver()
    if pcie_init() < 0 or pcie_open() < 0:
        print("❌ PCIe initialization failed")
        sys.exit(1)
    print("✓ PCIe device ready")
    
    # Storage for results across iterations
    cpu_fw_times = []
    cpu_bw_times = []
    fpga_fw_times = []
    fpga_bw_times = []
    accuracy_results = []
    
    try:
        for iteration in range(NUM_ITERATIONS):
            print(f"\n{'='*70}")
            print(f"Iteration {iteration + 1}/{NUM_ITERATIONS}")
            print(f"{'='*70}")
            
            # ====== Step 1: Send data via PCIe ======
            success, reference_output, ref_delta_x, ref_delta_weight, ref_delta_bias, cpu_fw_time, cpu_bw_time = send_all_data_via_pcie()
            if not success:
                print(f"\n❌ PCIe transfer failed in iteration {iteration + 1}, skipping")
                continue
            
            cpu_fw_times.append(cpu_fw_time)
            cpu_bw_times.append(cpu_bw_time)
            
            # Short delay for data stability
            time.sleep(0.1)
            
            # ====== Step 2: Activate FPGA via UDP ======
            if not send_activate_via_udp():
                print(f"\n❌ UDP communication failed in iteration {iteration + 1}, skipping")
                continue
            
            # ====== Step 3: Read and verify FPGA results ======
            # Modified to return timing and accuracy data
            result = read_and_compare_results(reference_output, ref_delta_x, ref_delta_weight, ref_delta_bias, cpu_fw_time, cpu_bw_time, iteration + 1, NUM_ITERATIONS)
            if result:
                fpga_fw_time, fpga_bw_time, accuracy = result
                fpga_fw_times.append(fpga_fw_time)
                fpga_bw_times.append(fpga_bw_time)
                accuracy_results.append(accuracy)
        
        # ====== Generate Average Report ======
        if len(cpu_fw_times) > 0 and len(fpga_fw_times) > 0:
            print(f"\n\n{'='*70}")
            print(f"{'RMSNorm Acceleration Circuit - Average Results':^70}")
            print(f"{'='*70}")
            print(f"\nTotal successful iterations: {len(fpga_fw_times)}/{NUM_ITERATIONS}")
            
            # Calculate averages
            avg_cpu_fw = np.mean(cpu_fw_times)
            avg_cpu_bw = np.mean(cpu_bw_times)
            avg_cpu_total = avg_cpu_fw + avg_cpu_bw
            
            avg_fpga_fw = np.mean(fpga_fw_times)
            avg_fpga_bw = np.mean(fpga_bw_times)
            avg_fpga_total = avg_fpga_fw + avg_fpga_bw
            
            # Accuracy statistics
            fw_pass_rate = sum(1 for acc in accuracy_results if acc['fw']) / len(accuracy_results) * 100
            grad_x_pass_rate = sum(1 for acc in accuracy_results if acc['grad_x']) / len(accuracy_results) * 100
            weight_pass_rate = sum(1 for acc in accuracy_results if acc['weight']) / len(accuracy_results) * 100
            
            print(f"CPU Type: Intel i5-10500\n")

            print(f"\n[Accuracy Statistics] (Threshold: rtol={RTOL}, atol={ATOL})")
            print(f"  Forward Output    : {fw_pass_rate:>6.1f}% pass rate")
            print(f"  Input Gradient    : {grad_x_pass_rate:>6.1f}% pass rate")
            print(f"  Weight Gradient   : {weight_pass_rate:>6.1f}% pass rate")
            
            print(f"\n[Average Performance]")
            print(f"{'':20} {'CPU (ms)':>14} {'FPGA (ms)':>14} {'Speedup':>12}")
            print(f"{'-'*70}")
            
            fw_speedup = avg_cpu_fw / avg_fpga_fw if avg_fpga_fw > 0 else 0
            bw_speedup = avg_cpu_bw / avg_fpga_bw if avg_fpga_bw > 0 else 0
            total_speedup = avg_cpu_total / avg_fpga_total if avg_fpga_total > 0 else 0
            
            print(f"  {'Forward':<18} {avg_cpu_fw:>14.5f} {avg_fpga_fw:>14.5f} {fw_speedup:>11.2f}x")
            print(f"  {'Backward':<18} {avg_cpu_bw:>14.5f} {avg_fpga_bw:>14.5f} {bw_speedup:>11.2f}x")
            print(f"{'-'*70}")
            print(f"  {'Total':<18} {avg_cpu_total:>14.5f} {avg_fpga_total:>14.5f} {total_speedup:>11.2f}x")
            
            print(f"{'='*70}")
            
            # Note about backward composition
            print(f"\nNote: Backward time = Weight Gradient + Input Gradient")
            print(f"\n✓ Average test completed successfully!")
        else:
            print("\n❌ No successful iterations to report")
        
        print("\n" + "="*70)
        print("All tests completed")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\nUser interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        pcie_close()


if __name__ == "__main__":
    main()

