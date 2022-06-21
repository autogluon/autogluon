function write_to_s3 {
    BUCKET="$1"
    DOC_PATH="$2"
    S3_PATH="$3"

    # Verify we still own the bucket
    bucket_query=$(aws s3 ls | grep -E "(^| )$BUCKET( |$)")
    if [ -n bucket_query ]; then
        if [ -d $DOC_PATH ]; then
            aws s3 cp --recursive $DOC_PATH $S3_PATH --quiet
        elif [ -f $DOC_PATH ]; then
            aws s3 cp $DOC_PATH $S3_PATH --quiet
        else
            echo Neither file nor directory being passed
        fi
    else
        echo Bucket does not belong to us anymore. Will not write to it
    fi
}