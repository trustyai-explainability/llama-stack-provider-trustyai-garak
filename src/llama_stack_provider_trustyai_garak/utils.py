from tqdm import tqdm
import time
from llama_stack_client import LlamaStackClient

def wait_for_completion_with_progress(client: LlamaStackClient, job_id: str, benchmark_id: str, poll_interval: int = 10):
    """Wait for job completion with tqdm progress bar"""
    
    def format_time(seconds: int | None, hours_needed: bool = True) -> str:
        """Format seconds to HH:MM:SS if hours_needed is True, otherwise MM:SS"""
        if seconds is None:
            return "N/A"
        if hours_needed:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            secs = seconds % 60
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes:02d}:{secs:02d}"
    
    status = client.alpha.eval.jobs.status(job_id=job_id, benchmark_id=benchmark_id)
    
    pbar = tqdm(
        total=100, 
        desc="Garak Scan",
        unit="%", 
        bar_format='{desc}: {n:.1f}%|{bar}| {postfix}',
        ncols=150
    )
    
    last_percent = 0
    
    while status.status in ["scheduled", "in_progress"]:
        progress = getattr(status, "metadata", {}).get("progress", {})
        
        current_percent = progress.get("percent", 0) if progress else 0
        
        #phase-1: Validation/Setup (no progress yet)
        if status.status in ["scheduled", "in_progress"] and not progress:
            pbar.set_description_str("Garak Setup")
            pbar.set_postfix_str("⚙️ Validating configuration and initializing Garak scan...")
        
        #phase-2: Scanning (progress available but < 100)
        elif progress and current_percent < 100:
            pbar.set_description_str("Garak Scan")
            
            # update overall progress bar
            pbar.update(max(0, current_percent - last_percent))
            last_percent = current_percent
            
            postfix_parts = []
            
            # overall scan times
            overall_elapsed = progress.get("overall_elapsed_seconds", 0)
            overall_eta = progress.get("overall_eta_seconds")
            
            if overall_eta:
                time_info = f"[{format_time(overall_elapsed)}<{format_time(overall_eta)}]"
            else:
                time_info = f"[{format_time(overall_elapsed)}]"
            
            postfix_parts.append(time_info)
            
            current_probe = progress.get("current_probe", "N/A")
            # shortn to save bar space
            probe_short = current_probe.replace("probes.", "").split('.')[-1]

            # overall metrics
            completed = int(progress.get("completed_probes", 0))
            total = int(progress.get("total_probes", 0))
            current_idx = min(completed + 1, total)
            postfix_parts.append(f"Current ({current_idx}/{total}): {probe_short}")
            
            #probe-specific (if available)
            if probe_progress := progress.get("current_probe_progress"):
                probe_pct = probe_progress.get("percent", 0)
                attempts_cur = probe_progress.get("attempts_current", 0)
                attempts_tot = probe_progress.get("attempts_total", 0)
                probe_eta_s = probe_progress.get("probe_eta_seconds")
                
                #probe-level detail
                if probe_eta_s is not None:
                    probe_eta_str = format_time(probe_eta_s, hours_needed=False).split(':')[-2:]
                    probe_eta_display = f"{probe_eta_str[0]}:{probe_eta_str[1]}"
                else:
                    probe_eta_display = "N/A"
                postfix_parts.append(
                    f"({probe_pct}%, Attempts: {attempts_cur}/{attempts_tot}, ETA:{probe_eta_display})"
                )
            
            pbar.set_postfix_str(" ".join(postfix_parts))
        
        #phase-3: Postprocessing (progress at 100% but job still in_progress)
        elif progress and current_percent >= 100 and status.status == "in_progress":
            if last_percent < 100:
                pbar.update(100 - last_percent)
                last_percent = 100
            pbar.set_description_str("Garak Scan Postprocessing")
            
            overall_elapsed = progress.get("overall_elapsed_seconds", 0)
            pbar.set_postfix_str(f"📊 Parsing results and uploading reports... [{format_time(overall_elapsed)}]")
        
        time.sleep(poll_interval)
        status = client.alpha.eval.jobs.status(job_id=job_id, benchmark_id=benchmark_id)
    
    pbar.n = 100
    pbar.close()
    if status.status in ['failed', 'completed', 'cancelled']:
        print("="*100)
        if status.status == 'completed':
            print(f"Job ended with status: {status.status} ✅")
        elif status.status == 'failed':
            print(f"Job ended with status: {status.status} ❌")
        elif status.status == 'cancelled':
            print(f"Job ended with status: {status.status} 🚫")
    return status