import numpy as np

def encrypt_message(message, key_matrix):
    message_vector = np.array([ord(char) for char in message]) # convert message to vector of ASCII values
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix) # get eigenvectors and eigenvalues of the key matrix
    diagonalized_key_matrix = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), np.linalg.inv(eigenvectors))  # diagonalize the key matrix
    encrypted_vector = np.dot(diagonalized_key_matrix, message_vector) # encrypt the message vector
    return encrypted_vector


def decrypt_message(encrypted_message, key_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix) # get eigenvectors and eigenvalues of the key matrix
    diagonalized_key_matrix = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), np.linalg.inv(eigenvectors)) # diagonalize the key matrix
    inverse_diagonalized_key_matrix = np.linalg.inv(diagonalized_key_matrix) # get the inverse of the diagonalized key matrix
    decrypted_vector = np.dot(inverse_diagonalized_key_matrix, encrypted_message) # decrypt the message vector
    decrypted_message = ''.join([chr(int(np.round(char.real))) for char in decrypted_vector])
    return decrypted_message


message = "Hello, World!"
key_matrix = np.random.randint(0, 256, (len(message), len(message)))

print("Original Message:", message)

encrypted_message = encrypt_message(message, key_matrix)
print("Encrypted message:", encrypted_message)

decrypted_message = decrypt_message(encrypted_message, key_matrix)
print("Decrypted message:", decrypted_message)
