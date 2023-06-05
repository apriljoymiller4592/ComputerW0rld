import cv2
import numpy
import os
import pygame

# Initialize Pygame and the display window
pygame.init()
size = (640, 480)
pygame.display.set_caption("Face Recognition")
screen = pygame.display.set_mode(size)

pygame.mixer.init()

# Replace 'your_audio_file.mp3' with your actual audio file
pygame.mixer.music.load('computerworld.mp3')

# Variables for face detection and recognition
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'
(width, height) = (130, 100)  # define constant size for all images
(images, labels, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = os.path.join(subjectpath, filename)
            label = id
            img = cv2.imread(path, 0)
            if img is not None:  # check if image is loaded successfully
                img = cv2.resize(img, (width, height))  # resize image
                images.append(img)
                labels.append(int(label))
        id += 1
(images, labels) = [numpy.array(lis) for lis in [images, labels]]

# Train the model
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

# Prepare for video capture and face detection
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)

# Initialize the font for caption
font = pygame.font.Font(None, 36)

# Define a button
button_color = (0, 204, 204)
button_pos = (size[0] - 350, size[1] - 50)
button_size = (80, 40)
button_rect = pygame.Rect(button_pos, button_size)

# Define caption
caption = "Welcome!"
text = font.render(caption, True, (255, 255, 255), (120, 120, 120))
text_rect = text.get_rect(center=(size[0] // 2, 20))  # put the text at the top center

# Music pause state
music_paused = False

# Main game loop
running = True
while running:
    # Check for QUIT event
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if button_rect.collidepoint(event.pos):
                if pygame.mixer.music.get_busy():
                    if pygame.mixer.music.get_pos() == 0:
                        pygame.mixer.music.play(-1)
                    elif pygame.mixer.music.get_pos() > 0 and not pygame.mixer.music.get_pos() >= pygame.mixer.music.get_endevent():
                        pygame.mixer.music.unpause()
                    else:
                        pygame.mixer.music.pause()
                else:
                    pygame.mixer.music.play(-1)

    (_, im) = webcam.read()
    resized_frame = cv2.resize(im, (100, 100))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))  # resize face
        prediction = model.predict(face_resize)
        if prediction[1] < 500:
            cv2.putText(im, '% s - %.0f' % (names[prediction[0]], prediction[1]), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)  # add this line to draw rectangle
        else:
            cv2.putText(im, 'not recognized', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)  # add this line to draw rectangle

    # flipping the image
    im = cv2.resize(im, size)
    im = cv2.flip(im, 1)

    # OpenCV to Pygame conversion
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = numpy.rot90(im)
    pygame.surfarray.blit_array(screen, im)

    # Create a Pygame surface from the image
    im_surface = pygame.surfarray.make_surface(im)

    # Position where the webcam frame will start in the pygame window
    frame_position = (10, 10)

    # Draw the image onto the screen
    screen.blit(im_surface, frame_position)

    # Draw the button
    pygame.draw.rect(screen, button_color, button_rect)

    # Draw the text
    screen.blit(text, text_rect)

    # Update the display
    pygame.display.flip()

pygame.mixer.music.stop()
pygame.quit()
