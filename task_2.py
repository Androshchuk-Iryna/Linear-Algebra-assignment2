import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the image
image = imread('/Users/mac/PycharmProject/image/photo_2024-06-15 14.09.05.jpeg')


# Convert the image to black and white
def bw_image(image):
    image_sum = image.sum(axis=2)
    image_bw = image_sum / image_sum.max()
    return image_bw

image_bw = bw_image(image)

fig, axs = plt.subplots(1,2 , figsize=(10, 5))
axs[0].imshow(image)
axs[0].set_title('Original Image')
axs[1].imshow(image_bw, cmap='gray')
axs[1].set_title('Black and White Image')

print("Original image dimensions:", image.shape)
print("Number of color channels:", image.shape[2])
print("Black and white image dimensions:", image_bw.shape)
plt.show()
#-----------------------------------------------------------------------------------------------------------------------


pca = PCA()


def pca_image(image_bw,pca):
    image_bw_pca = pca.fit_transform(image_bw)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    threshold = 0.95
    num_components_index = np.where(cumulative_variance >= threshold)[0][0]
    num_components = num_components_index + 1
    return image_bw_pca, num_components, cumulative_variance

com_image_bw_pca, num_components, cumulative_variance = pca_image(image_bw, pca)
print("Кількість компонент, які необхідні для покриття 95% варіативності:", num_components)


plt.figure(figsize=(10, 7))
plt.plot(cumulative_variance, linestyle='-', color='b')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.axvline(x=num_components, color='g', linestyle='--')
plt.plot(cumulative_variance)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Components')
plt.show()
plt.grid()

# ----------------------------------------------------------------------------------------------------------------------

def reconstruct_image(image_bw_pca, num_components):
    pca = PCA(n_components=num_components)
    pca.fit(image_bw)
    image_bw_pca_limited = image_bw_pca[:, :num_components]
    image_reconstructed = pca.inverse_transform(image_bw_pca_limited)
    return image_reconstructed

image_reconstructed_80 = reconstruct_image(com_image_bw_pca, 9)
image_reconstructed_90 = reconstruct_image(com_image_bw_pca, 19)
image_reconstructed_95 = reconstruct_image(com_image_bw_pca, 41)
image_reconstructed_96 = reconstruct_image(com_image_bw_pca, 50)
image_reconstructed_97 = reconstruct_image(com_image_bw_pca, 63)
image_reconstructed_98 = reconstruct_image(com_image_bw_pca, 85)
image_reconstructed_99 = reconstruct_image(com_image_bw_pca, 126)
image_reconstructed = reconstruct_image(com_image_bw_pca, 455)



fig, axs = plt.subplots(3, 3, figsize=(10, 10))

axs[0, 0].imshow(image_reconstructed_80, cmap='gray')
axs[0, 0].set_title('80% Variance')
axs[0, 1].imshow(image_reconstructed_90, cmap='gray')
axs[0, 1].set_title('90% Variance')
axs[0, 2].imshow(image_reconstructed_95, cmap='gray')
axs[0, 2].set_title('95% Variance')
axs[1, 0].imshow(image_reconstructed_96, cmap='gray')
axs[1, 0].set_title('96% Variance')
axs[1, 1].imshow(image_reconstructed_97, cmap='gray')
axs[1, 1].set_title('97% Variance')
axs[1, 2].imshow(image_reconstructed_98, cmap='gray')
axs[1, 2].set_title('98% Variance')
axs[2, 0].imshow(image_reconstructed_99, cmap='gray')
axs[2, 0].set_title('99% Variance')
axs[2, 1].imshow(image_reconstructed, cmap='gray')
axs[2, 1].set_title('99.99% Variance')
axs[2, 2].imshow(image_bw, cmap='gray')
axs[2, 2].set_title('Original Image')

plt.tight_layout()
plt.show()





