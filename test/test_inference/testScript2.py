def log_directory_structure(directory_path, ai_context, indent=0):
    # Check if path is valid
    if not os.path.exists(directory_path):
        print("Directory not found.")
        return

    # Get the list of items in the directory
    items = os.listdir(directory_path)

    for item in items:
        item_path = os.path.join(directory_path, item)

        # Check if the item is a directory
        if os.path.isdir(item_path):
            # Log the directory using ai_context
            directory_name = os.path.basename(item_path)
            indentation = "  " * indent  # Adjust indentation for subdirectories
            log_message = f"{indentation}Directory: {directory_name}"
            print(log_message)

            # Recursively log the subdirectory structure
            log_directory_structure(item_path, ai_context, indent + 1)

        # If it's a file, you can log it similarly

    # Get the directory path from the user
    directory_path = input("Enter the directory path: ")

    # Log the root directory
    print(f"Root Directory: {directory_path}")

    # Call the function to log the directory structure
    log_directory_structure(directory_path, ai_context)

    print(ai_context)