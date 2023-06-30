import os

import boto3

# boto3 session setup
session = boto3.Session(profile_name="autogluon_mxnet_io_ci_operator")
s3 = session.client("s3")

# S3 identifiers
bucket_name = "autogluon.mxnet.io"
versions = ["0.4.0"]  # I will run one version to test the script, then I can run the rest
# versions = ['0.0.14', '0.0.15', '0.1.0', '0.2.0', '0.3.0', '0.3.1', '0.4.0', '0.4.1', '0.4.2', '0.4.3', '0.5.1', '0.5.2', '0.5.3', '0.6.0', '0.6.1', '0.6.2', '0.7.0']

old_analytics_identifier = "UA-96378503-20"
new_analytics_identifier = "G-6XDS99SP0C"

for version in versions:
    # List all the objects in the S3 bucket
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=version)

    # Iterate over the objects and process the HTML files
    for obj in response["Contents"]:
        key = obj["Key"]
        if key.endswith(".html"):
            original_content_filename = os.path.join("original", version, key)
            modified_content_filename = os.path.join("modified", version, key)

            # Download the HTML file
            s3.download_file(bucket_name, key, original_content_filename)

            # Open the downloaded file
            with open(original_content_filename, "r") as original_content_fo:
                original_content = original_content_fo.read()

            # Replace the substring
            modified_content = original_content.replace(old_analytics_identifier, new_analytics_identifier)

            # Save the modified file
            with open(modified_content_filename, "w") as modified_content_fo:
                modified_content_fo.write(modified_content)

            # Upload the modified file back to the S3 bucket
            s3.upload_file(modified_content_filename, bucket_name, key)
