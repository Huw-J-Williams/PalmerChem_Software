## script to check slurm q to see whether a specified list of jobs have finished

# %%
import subprocess
import time
import logging

def WaitForJobs(job_id_ls: list, username: str, wait_time: int = 60):
	'''Set up logger. Read squeue output, check for jobs. If still running, wait and check again.'''

	##################### set up logger ######################
	
    logging.basicConfig(
        format= "%(asctime)s  - %(levelname)s - %(message)s",
        level = logging.INFO
    )

	################ loop to check q and wait ################
 	 
	## check queue
    while True:
        dock_jobs = []

        try:
            squeue = subprocess.check_output(["squeue", "--users", username], text=True)

        except subprocess.CalledProcessError as e:
            logging.error(f"Error excecuting squeue command: {e}")
            return False

		## read output and check for jobs in list
        lines = squeue.splitlines() # split q output into lines (equivalent to jobs)
        job_lines = {line.split()[0]: line for line in lines if len(line.split()) > 0} # create dictionary of job_no : line, for populated lines

        dock_jobs = set(job_id for job_id in job_id_ls if job_id in job_lines) # collect job_ids that are in list of interst and also in q

		## set up output for user
        if not dock_jobs: # if none of specified jobs are in q...
            logging.info("All jobs have completed.")
            return False

        if len(dock_jobs) < 10: # if only a few specified jobs are in q...
            job_message = (
                f'Waiting for the following jobs to complete: {", ".join(dock_jobs)}'
            )

        else: # if non of specified jobs found in q...
            job_message = f"Waiting for {len(dock_jobs)} jobs to finish"

        logging.info(job_message) # print message
        time.sleep(wait_time) # wait for user defined time before chekcing again
