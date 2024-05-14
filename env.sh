# Set the name of the virtual environment
ENV_NAME="my_venv"

# Check if the virtual environment directory exists
if [ -d "$ENV_NAME" ]; then
    echo "Virtual environment '$ENV_NAME' already exists. Activating it..."
else
    # Create the virtual environment if it does not exist
    echo "Creating virtual environment '$ENV_NAME'..."
    python3 -m venv $ENV_NAME
fi

# Activate the virtual environment
source $ENV_NAME/bin/activate

# Check if requirements.txt exists and install dependencies
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

# Notify the user
echo "Virtual environment '$ENV_NAME' is active!"