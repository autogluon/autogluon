import argparse
import re

import boto3

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--profile", help="profile name of aws account.", type=str, default=None)
parser.add_argument("--region", help="Default region when creating new connections", type=str, default="us-east-1")
parser.add_argument("--name", help="name of the job to be cancelled.", type=str, default="")
parser.add_argument("--job-type", help="name of the job to be cancelled.", type=str, default="CI-CPU")
parser.add_argument(
    "--reason",
    help="reason to cancel the job.",
    type=str,
    default="Canelling because new commits related to same PR is pushed",
)
args = parser.parse_args()

profile = args.profile
region = args.region
job_name = re.sub("[^A-Za-z0-9_\-]", "", args.name)[:128]  # Enforce AWS Batch jobName rules
job_type = args.job_type
reason = args.reason

session = boto3.Session(profile_name=profile, region_name=region)
batch = session.client(service_name="batch")


def main():
    # Find all jobs with job_name that are running or about to run and terminate them all.
    list_args = {
        "jobQueue": job_type,
        "filters": [{"name": "JOB_NAME", "values": [job_name]}],
    }
    try:
        response = batch.list_jobs(**list_args)
        for job in response["jobSummaryList"]:
            if job["status"] in ["SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING"]:
                print(f"Terminate previous job {job['jobId']}")
                batch.terminate_job(jobId=job["jobId"], reason="New job submitted")
    except Exception as e:
        print(f"Failed to terminate the job because of exception: {e}")


if __name__ == "__main__":
    main()
