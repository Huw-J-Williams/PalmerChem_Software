
%%
import subprocess #Automation, used to run other code
import time #Allows for control of time based functions, used to delay the programme
import logging #Helps to see programme bugs and crashes

"""
Function that checks queue for SLURM jobs submitted by user. It repeatedly checks whether these jobs are still running and waits until all of them are completed before exiting.

Parameters:

job_id_ls (list)    A list of job IDs to monitor.

username (str)      The username of the person that submitted the job.

wait_time (int)     The time (in seconds) to wait between job status checks.

"""

#Creates function WaitForJobs, which monitors submitted jobs and waits until all jobs are complete
def WaitForJobs(job_id_ls: list, username: str, wait_time: int = 60):
#Setting the format for how logging information is formatted and sets the logging level to INFO
logging.basicConfig(
    format= "%(asctime)s  - %(levelname)s - %(message)s",
    level = logging.INFO
)

while True:
		#Saving running docking jobs to list
    dock_jobs = []
		#Checks squeue for job submitted by user, if no job it logs this and stops function
    try:
        squeue = subprocess.check_output(["squeue", "--users", username], text=True)

    except subprocess.CalledProcessError as e:
        logging.error(f"Error excecuting squeue command: {e}")
        return False
    #Splitting list of jobs into separate lines, to be able to run each job ID as its own job
    lines = squeue.splitlines()
    #Saving those split up jobs to a dictionary
    job_lines = {line.split()[0]: line for line in lines if len(line.split()) > 0}

		#Updates the job lists with only running jobs
    dock_jobs = set(job_id for job_id in job_id_ls if job_id in job_lines)
		
		#If no jobs left to run, logs completion message and finishes programme
    if not dock_jobs:
        logging.info("All jobs have completed.")
        return False
        
		#Checks number of docking jobs completed, if less than 10, logs remaining jobs, otherwise shows exact number of jobs left to complete
    if len(dock_jobs) < 10:
        job_message = (
            f'Waiting for the following jobs to complete: {", ".join(dock_jobs)}'
        )

    else:
        job_message = f"Waiting for {len(dock_jobs)} jobs to finish"
    
    #Logs current jobs and time.sleep - delays running the programme for the 60 seconds specified
    logging.info(job_message)
    time.sleep(wait_time)
