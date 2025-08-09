import os

def update_env_file(key, value):
    """Updates or adds a key-value pair to the .env file."""
    env_file = '.env'
    if not os.path.exists(env_file):
        print(f"Warning: {env_file} not found. Cannot save {key}.")
        return

    with open(env_file, 'r') as f:
        lines = f.readlines()

    key_found = False
    with open(env_file, 'w') as f:
        for line in lines:
            if line.strip().startswith(f'{key}='):
                f.write(f'{key}={value}\n')
                key_found = True
            else:
                f.write(line)
        
        if not key_found:
            f.write(f'\n{key}={value}\n')
    
    print(f"✓ Successfully saved {key}={value} to .env file.")