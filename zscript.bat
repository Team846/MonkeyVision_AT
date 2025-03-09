setlocal

set SOURCE_DIR=X:\Vision\MonkeyVision_AT\*
set TARGET_USER=orangepi
set TARGET_HOST=funkyvision1.local
set TARGET_DIR=/home/orangepi/MonkeyVision_AT/

scp -P 5806 -r "%SOURCE_DIR%" "%TARGET_USER%@%TARGET_HOST%:%TARGET_DIR%" 

echo Process completed
endlocal