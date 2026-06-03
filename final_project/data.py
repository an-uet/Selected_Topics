import kagglehub

# Download latest version
path = kagglehub.competition_download('super-resolution-in-video-games')

print("Path to competition files:", path)