AZURE_STORAGE_ACCESS_KEY=NmU7iRjY5IrCiWVlYV0+Gja9PdnflTgzEOtffYXhfC1Ko4qtjqXcKcJ3Be1RYNs4k/QH5gEwcOcpjWniyD3PgQ==

CLUSTER=sc3
VC=resrchvc
JOB_NAME=Megatron-LM_test
USER=t-yuniu
CONFIG_FILE=/blob/Megatron-LM/philly_run.sh
NUM_GPUS=4
NUM_CONTAINERS=1
DEBUG=false
RES=$(curl -s -H "Content-Type: application/json" \
            -H "WWW-Authenticate: Negotiate" \
            -H "WWW-Authenticate: NTLM" \
            -X POST https://philly/api/v2/submit -k --ntlm -n -d \
            "{
                \"UserName\": \"$USER\",
                \"Inputs\": [{
                    \"Path\": \"/blob\",
                    \"Name\": \"data\",
                }],
                \"queue\":\"\",
                \"IsCrossRack\": false,
                \"IsMemCheck\": false,
                \"RackId\": \"anyConnected\",
                \"MinGPUs\": \"$NUM_GPUS\",
                \"ToolType\": null,
                \"BuildId\": 0,
                \"Outputs\": [],
                \"ClusterId\": \"$CLUSTER\",
                \"IsDebug\": $DEBUG,
                \"JobName\": \"$JOB_NAME\",
                \"ConfigFile\": \"$CONFIG_FILE\",
                \"Tag\": \"v1.4-py36-cuda10\",
                \"Repository\": \"philly/jobs/custom/pytorch-tts\",
                \"PrevModelPath\": null,
                \"Registry\": \"phillyregistry.azurecr.io\",
                \"VcId\": \"$VC\",
                \"SubmitCode\": \"p\",
                \"dynamicContainerSize\": false,
                \"OneProcessPerContainer\": true,
                \"volumes\": {
                     \"myblob\": {
                        \"type\": \"blobfuseVolume\",
                        \"storageAccount\": \"chatphilly\",
                        \"containerName\": \"yuniu\",
                        \"path\": \"/blob\"
                     },
                },
                \"credentials\": {
                    \"storageAccounts\": {
                        \"chatphilly\": {
                            \"key\": \"$AZURE_STORAGE_ACCESS_KEY\"
                        }
                    }
                },
                \"NumOfContainers\": $NUM_CONTAINERS,
                \"CustomMPIArgs\": \"env OMPI_MCA_BTL=self,sm,tcp,openib\",
                \"Timeout\": null
            }")

echo $RES
