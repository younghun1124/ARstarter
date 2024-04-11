import numpy as np
import cv2 as cv

def select_img_from_video(video_file, board_pattern, select_all=False, wait_msec=10, wnd_name='Camera Calibration'):
    # Open a video
    video = cv.VideoCapture(video_file)
    assert video.isOpened()

    # Select images
    img_select = []
    while True:
        # Grab an images from the video
        valid, img = video.read()
        if not valid:
            break

        if select_all:
            img_select.append(img)
        else:
            # Show the image
            display = img.copy()
            cv.putText(display, f'NSelect: {len(img_select)}', (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
            cv.imshow(wnd_name, display)

            # Process the key event
            key = cv.waitKey(wait_msec)
            if key == ord(' '):             # Space: Pause and show corners
                complete, pts = cv.findChessboardCorners(img, board_pattern)
                cv.drawChessboardCorners(display, board_pattern, pts, complete)
                cv.imshow(wnd_name, display)
                key = cv.waitKey()
                if key == ord('\r'):
                    img_select.append(img) # Enter: Select the image
            if key == 27:                  # ESC: Exit (Complete image selection)
                break

    cv.destroyAllWindows()
    return img_select

def calib_camera_from_chessboard(images, board_pattern, board_cellsize, K=None, dist_coeff=None, calib_flags=None):
    # Find 2D corner points from given images
    img_points = []
    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        complete, pts = cv.findChessboardCorners(gray, board_pattern)
        if complete:
            img_points.append(pts)
    assert len(img_points) > 0

    # Prepare 3D points of the chess board
    obj_pts = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
    obj_points = [np.array(obj_pts, dtype=np.float32) * board_cellsize] * len(img_points) # Must be `np.float32`

    # Calibrate the camera
    return cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], K, dist_coeff, flags=calib_flags)

def disortion_correction(video_file,K,dist_coeff):
    
# Open a video
    video = cv.VideoCapture(video_file)
    assert video.isOpened(), 'Cannot read the given input, ' + video_file

    # Run distortion correction
    show_rectify = True
    map1, map2 = None, None
    while True:
        # Read an image from the video
        valid, img = video.read()
        if not valid:
            break

        # Rectify geometric distortion (Alternative: `cv.undistort()`)
        info = "Original"
        if show_rectify:
            if map1 is None or map2 is None:
                map1, map2 = cv.initUndistortRectifyMap(K, dist_coeff, None, None, (img.shape[1], img.shape[0]), cv.CV_32FC1)
            img = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR)
            info = "Rectified"
        cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

        # Show the image and process the key event
        cv.imshow("Geometric Distortion Correction", img)
        key = cv.waitKey(10)
        if key == ord(' '):     # Space: Pause
            key = cv.waitKey()
        if key == 27:           # ESC: Exit
            break
        elif key == ord('\t'):  # Tab: Toggle the mode
            show_rectify = not show_rectify

    
    video.release()
    cv.destroyAllWindows()


if __name__ == '__main__': 
    video_file = 'chessboard.mp4'
    board_pattern = (10, 7)
    board_cellsize = 0.025
    
    img_select = select_img_from_video(video_file, board_pattern)
    assert len(img_select) > 0, 'There is no selected images!'
    rms, K, dist_coeff, rvecs, tvecs = calib_camera_from_chessboard(img_select, board_pattern, board_cellsize)
    board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK
    # Open a video
    video = cv.VideoCapture(video_file)
    assert video.isOpened(), 'Cannot read the given input, ' + video_file

    # Prepare a 3D box for simple AR
    box_lower = board_cellsize * np.array([[4, 2,  0], [5, 2,  0], [5, 4,  0], [4, 4,  0]])
    box_upper = board_cellsize * np.array([[4, 2, -1], [5, 2, -1], [5, 4, -1], [4, 4, -1]])

    # Prepare 3D points on a chessboard
    obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

    # Run pose estimation
    while True:
        # Read an image from the video
        valid, img = video.read()
        if not valid:
            break

        # Estimate the camera pose
        success, img_points = cv.findChessboardCorners(img, board_pattern, board_criteria)
        if success:
            ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

            # Draw the box on the image
            line_lower, _ = cv.projectPoints(box_lower, rvec, tvec, K, dist_coeff)
            line_upper, _ = cv.projectPoints(box_upper, rvec, tvec, K, dist_coeff)
            cv.polylines(img, [np.int32(line_lower)], True, (255, 0, 0), 2)
            cv.polylines(img, [np.int32(line_upper)], True, (0, 0, 255), 2)

            # Increase the height of the box
            box_lower_higher = board_cellsize * np.array([[4, 2,  1], [5, 2,  1], [5, 4,  1], [4, 4,  1]])
            box_upper_higher = board_cellsize * np.array([[4, 2,  0], [5, 2,  0], [5, 4,  0], [4, 4,  0]])

            line_lower_higher, _ = cv.projectPoints(box_lower_higher, rvec, tvec, K, dist_coeff)
            line_upper_higher, _ = cv.projectPoints(box_upper_higher, rvec, tvec, K, dist_coeff)
            cv.polylines(img, [np.int32(line_lower_higher)], True, (255, 0, 0), 2)
            cv.polylines(img, [np.int32(line_upper_higher)], True, (0, 0, 255), 2)

            for b, t in zip(line_lower, line_upper):
                cv.line(img, np.int32(b.flatten()), np.int32(t.flatten()), (0, 255, 0), 2)
            for b, t in zip(line_lower_higher, line_upper_higher):
                cv.line(img, np.int32(b.flatten()), np.int32(t.flatten()), (0, 255, 0), 2)

            # Create boxes on the left and right sides
            box_left_lower = board_cellsize * np.array([[3, 2,  0], [4, 2,  0], [4, 4,  0], [3, 4,  0]])
            box_left_upper = board_cellsize * np.array([[3, 2, -1], [4, 2, -1], [4, 4, -1], [3, 4, -1]])

            box_right_lower = board_cellsize * np.array([[5, 2,  0], [6, 2,  0], [6, 4,  0], [5, 4,  0]])
            box_right_upper = board_cellsize * np.array([[5, 2, -1], [6, 2, -1], [6, 4, -1], [5, 4, -1]])

            line_left_lower, _ = cv.projectPoints(box_left_lower, rvec, tvec, K, dist_coeff)
            line_left_upper, _ = cv.projectPoints(box_left_upper, rvec, tvec, K, dist_coeff)
            cv.polylines(img, [np.int32(line_left_lower)], True, (255, 0, 0), 2)
            cv.polylines(img, [np.int32(line_left_upper)], True, (0, 0, 255), 2)

            line_right_lower, _ = cv.projectPoints(box_right_lower, rvec, tvec, K, dist_coeff)
            line_right_upper, _ = cv.projectPoints(box_right_upper, rvec, tvec, K, dist_coeff)
            cv.polylines(img, [np.int32(line_right_lower)], True, (255, 0, 0), 2)
            cv.polylines(img, [np.int32(line_right_upper)], True, (0, 0, 255), 2)

            for b, t in zip(line_left_lower, line_left_upper):
                cv.line(img, np.int32(b.flatten()), np.int32(t.flatten()), (0, 255, 0), 2)
            for b, t in zip(line_right_lower, line_right_upper):
                cv.line(img, np.int32(b.flatten()), np.int32(t.flatten()), (0, 255, 0), 2)

            # Print the camera position
            R, _ = cv.Rodrigues(rvec) # Alternative) `scipy.spatial.transform.Rotation`
            p = (-R.T @ tvec).flatten()
            info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
            cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

        # Show the image and process the key event
        cv.imshow('Pose Estimation (Chessboard)', img)
        key = cv.waitKey(10)
        if key == ord(' '):
            key = cv.waitKey()
        if key == 27: # ESC
            break

    video.release()
    cv.destroyAllWindows()
    
    
    
    # Print calibration results
    print('## Camera Calibration Results')
    print(f'* The number of selected images = {len(img_select)}')
    print(f'* RMS error = {rms}')
    print(f'* Camera matrix (K) = \n{K}')
    print(f'* Distortion coefficient (k1, k2, p1, p2, k3, ...) = {dist_coeff.flatten()}')