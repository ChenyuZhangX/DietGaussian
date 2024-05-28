from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary

if __name__ == "__main__":
    path = "cameras/toy/001.bin"
    # read .bin file show the content
    with open("cameras/toy/001.bin", "rb") as file:
        # Read the entire file
        data = file.read()

    # Print the entire file
    print(data)
    # Convert each byte to a binary string and join them together
    data_binary = "".join(f"{byte:08b}" for byte in data)

    # Print the binary data
    print(data_binary.decode("utf-8"))
