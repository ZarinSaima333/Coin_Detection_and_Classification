import cv2
import numpy as np

# Initialize webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Get screen dimensions
screen_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
screen_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define big square box in the middle of the screen
box_size = 200  # Size of the square box
box_x1 = (screen_width - box_size) // 2
box_y1 = (screen_height - box_size) // 2
box_x2 = box_x1 + box_size
box_y2 = box_y1 + box_size

print(f"Screen dimensions: {screen_width}x{screen_height}")
print(f"Box coordinates: ({box_x1}, {box_y1}) to ({box_x2}, {box_y2})")

while True:
    # Read frame from camera
    ret, frame = cap.read()
    if not ret:
        break
    
    output = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Reduce noise
    gray = cv2.medianBlur(gray, 5)
    
    # Detect circles - using the same parameters you provided
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=1000,      
        param1=100,         
        param2=40,       
        minRadius=20,       
        maxRadius=100
    )
    
    # 1. Draw a big square box in the middle of the screen
    cv2.rectangle(output, (box_x1, box_y1), (box_x2, box_y2), (255, 0, 0), 2)
    
    # Initialize message variable
    message = ""
    
    # Process detected circles
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            x, y, r = circle
            
            # Draw circle
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)  # Circle outline
            cv2.circle(output, (x, y), 2, (0, 0, 255), 3)  # Center point
            
            # 2. Get the position of the circle in the console
            print(f"Circle position: ({x}, {y}), Radius: {r}")
            
            # Calculate circle area and box area
            circle_area = np.pi * r * r
            box_area = box_size * box_size
            coverage_percentage = (circle_area / box_area) * 100
            
            print(f"Circle coverage: {coverage_percentage:.2f}%")
            
            # 3. Check if circle center is in the box
            circle_in_box = (box_x1 <= x <= box_x2) and (box_y1 <= y <= box_y2)
            
            if circle_in_box:
                if coverage_percentage >= 50:
                    message = "Click on the screen"
                    print("Click on the screen")
                else:
                    message = "Move closer"
                    print("Move closer")
            else:
                # 4. Determine position relative to box and print instructions
                if y > box_y2:  # Below the box
                    message = "Move up"
                    print("Move up")
                elif y < box_y1:  # Above the box
                    message = "Move down"
                    print("Move down")
                elif x > box_x2:  # Right of the box
                    message = "Move right"
                    print("Move right")
                elif x < box_x1:  # Left of the box
                    message = "Move left"
                    print("Move left")
    
    # Display message on the screen
    if message:
        cv2.putText(output, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Show coverage percentage on screen if circle is detected
    if circles is not None and 'coverage_percentage' in locals():
        coverage_text = f"Coverage: {coverage_percentage:.1f}%"
        cv2.putText(output, coverage_text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Show result
    cv2.imshow('Real-Time Circle Detection', output)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
