
# Downloading data from Physionet servers
echo "Downloading data from Physionet servers ..."
wget https://physionet.org/static/published-projects/qtdb/qt-database-1.0.0.zip -OutFile qt-database-1.0.0.zip
wget https://physionet.org/static/published-projects/nstdb/mit-bih-noise-stress-test-database-1.0.0.zip -OutFile mit-bih-noise-stress-test-database-1.0.0.zip

# Create a data folder
mkdir data

# Extract data
echo "Extracting data ..."
Expand-Archive qt-database-1.0.0.zip -DestinationPath .
Expand-Archive mit-bih-noise-stress-test-database-1.0.0.zip -DestinationPath .

# Move folders inside data
echo "Moving folders inside data..."
mv qt-database-1.0.0 data/
mv mit-bih-noise-stress-test-database-1.0.0 data/

# Remove zip files
echo "Removing zip files..."
rm qt-database-1.0.0.zip
rm mit-bih-noise-stress-test-database-1.0.0.zip

echo "Done."