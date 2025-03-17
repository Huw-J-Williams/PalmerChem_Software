# A program which checks which jobs in a list of slurm jobs (IDs) are running, and returns the jobs that are (every 60 seconds by default)

# %%
#* Module Imports
import subprocess #? For access to commands on the command-line
import time #? For waiting X seconds between checks
import logging #? For logging

def WaitForJobs(job_id_ls: list, username: str, wait_time: int = 60): #* Checks a list of job IDs (job_id_ls) to see which jobs in the list are still running on Slurm under a specific username, outputting the ones that are every wait_time seconds until all the jobs have been completed
    """
    Checks to see which Slurm jobs are running

    Summary
    -------
    Checks a list of job IDs (job_id_ls) to see which jobs in the list are still running on Slurm for a specific user.
    Any jobs that are still running will be logged - if more than 10 jobs are running, only the *number* of running jobs
    will be output. This check is looped every wait_time seconds until all specified jobs have finished running

    Parameters
    ----------
    job_id_ls : list
        The list of IDs to be monitored until completion
    username : str
        The username for the account that the jobs originate from, in the format `xxx00000`
    wait_time : int, optional
        The time, in seconds, to wait between checks (default 60)
    
    Returns
    -------
    bool
        False

    Raises
    ------
    CalledProcessError
        If the call to Slurm fails

    """

    logging.basicConfig(
        format= "%(asctime)s  - %(levelname)s - %(message)s",
        level = logging.INFO
    ) # Sets up 

    while True: # While there's at least one job still running
        dock_jobs = [] #? The list for jobs

        try:
            squeue = subprocess.check_output(["squeue", "--users", username], text=True) # Makes a call to the slurm queue, which will return all jobs under the specified username

        except subprocess.CalledProcessError as e: # If the call to the queue fails, log the error and exit the function
            logging.error(f"Error excecuting squeue command: {e}")
            return False
        
        lines = squeue.splitlines() # Splits the queue output into individual jobs (each line is a job)
        job_lines = {line.split()[0]: line for line in lines if len(line.split()) > 0} # Adds each job to a dictionary in the format {<the ID of the job>: <the job>}

        dock_jobs = set(job_id for job_id in job_id_ls if job_id in job_lines) # Checks current job IDs for the user against the input list of job IDs. If any IDs are also in the queue call (i.e. the job is running), then add that ID to the set of IDs 

        if not dock_jobs: # If there are no jobs in the set of job ids (i.e. no active jobs), then there are no (specified) jobs running - log a success and return False
            logging.info("All jobs have completed.")
            return False

        if len(dock_jobs) < 10: # If there are fewer than 10 active jobs, log all the IDs that are still running
            job_message = (
                f'Waiting for the following jobs to complete: {", ".join(dock_jobs)}'
            )

        else: # If there are more than 10 active jobs, log the number of jobs running (do not report the individual IDs) 
            job_message = f"Waiting for {len(dock_jobs)} jobs to finish"

        logging.info(job_message) # Logs the message that some number of jobs are running
        time.sleep(wait_time) # Pause the program for the specified number of seconds, before looping through (checking whether the job IDs are still running) again

# %%

# How to run the code (will need to add in a job ID and your own username)

WaitForJobs(job_id_ls=[""], username="", wait_time=60)

