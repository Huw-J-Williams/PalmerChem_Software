# %%
import subprocess
import time
import logging

def WaitForJobs(job_id_ls: list, username: str, wait_time: int = 60):

    logging.basicConfig(
        format= "%(asctime)s  - %(levelname)s - %(message)s",
        level = logging.INFO
    )

    while True:
        dock_jobs = []

        try:
            squeue = subprocess.check_output(["squeue", "--users", username], text=True)

        except subprocess.CalledProcessError as e:
            logging.error(f"Error excecuting squeue command: {e}")
            return False
        
        lines = squeue.splitlines()
        job_lines = {line.split()[0]: line for line in lines if len(line.split()) > 0}

        dock_jobs = set(job_id for job_id in job_id_ls if job_id in job_lines)

        if not dock_jobs:
            logging.info("All jobs have completed.")
            return False

        if len(dock_jobs) < 10:
            job_message = (
                f'Waiting for the following jobs to complete: {", ".join(dock_jobs)}'
            )

        else:
            job_message = f"Waiting for {len(dock_jobs)} jobs to finish"

        logging.info(job_message)
        time.sleep(wait_time)
