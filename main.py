#==============================================================================
    # Parameters
        #scale exponent (for fast processing)
        #Capture source
#
    # Take Initial Video Capture
#
    # Capture 2 Frames
# 
    # Convert to Gray
# 
    # Transpose Size 
# 
# Resize 3x
# 
    # Perform optical flow
# 
# Denoise: 3avg and thresh
# 
# Point transpose
# Slice index combine
# 
# Final resize
# 
   # Revolution
#
#==============================================================================
import cv2
import numpy as np

# Parameters

res = .5

thresh = 1.

weight = 12
def weight():
    None

cap = cv2.VideoCapture(0)

# Take Initial Capture

prev = cap.read()[1]

while 1:
    # Capture 2 Frames

    next = cap.read()[1]
    
    total_width = next.shape[1]
    total_height = next.shape[0]

    
    # Convert to Gray
    
    prev_gray = cv2.cvtColor(prev,cv2.COLOR_RGB2GRAY)
    next_gray = cv2.cvtColor(next,cv2.COLOR_RGB2GRAY)
    
    # Scale Down Working Size

    prev_small = cv2.resize(prev_gray, (0,0), fx = res, fy = res)
    next_small = cv2.resize(next_gray, (0,0), fx = res, fy = res)    
    
    # Perform Optical Flow
        #between prev and next
    flow = cv2.calcOpticalFlowFarneback(    \
        prev_small, \
        next_small, \
        None,   \
        #pyr_scale
        .5,     \
        #levels
        3,      \
        #winsize
        5,      \
        #iterations
        3,      \
        #poly_n
        9,      \
        #
        1.5,    \
        #
        cv2.OPTFLOW_FARNEBACK_GAUSSIAN)      \

        #separate the x and y flows
    _flow_x = flow[:,:,0]
    _flow_y = flow[:,:,1]
    
        #resize x and y flow arrays to original size
        #
        #improve this variety
        #
    flow_x_ = cv2.resize(_flow_x, (0,0), fx = 1/res, fy = 1/res)    
    flow_y_ = cv2.resize(_flow_y, (0,0), fx = 1/res, fy = 1/res)
   # flow_s
    
    #Point Transpose: Index Slice/Combine
        #delineate flow arrays
            #format and weight transform
    flow_x_linear = flow_x_.ravel().astype(np.int)*weight
    flow_y_linear = flow_y_.ravel().astype(np.int)*weight
        
        #create a 1d array of the same flat length    
    increase = np.arange(flow_y_linear.shape[0])
    
        #modulate and stack
    mod_x = increase % total_width
    mod_y = (increase/total_width).astype(np.int) 
    
    
#   mod_x                         mod_y
#    
#   -----~~~~~=====OOOOO00000     -~=O0-~=O0-~=O0-~=O0-~=O0


        #join x and y mods
    mod_y_x = np.stack((mod_y, mod_x))

    
    shift_x = mod_x+flow_x_linear
    shift_y = mod_y+flow_y_linear
    
            #remove outliers    
    shift_x[(shift_x>total_width-10) & (shift_x<-1*total_width-10)]=0

    shift_y[(shift_y>total_height-10) & (shift_y<-1*total_height-10)]=0
    
    shift_y_x = np.stack((shift_y, shift_x))
    
    #Revolve
    
    prev = next.copy()
    super = next.copy()
    
    super[(shift_y_x[0,:],shift_y_x[1,:])]  \
        =                                   \
        super[(mod_y_x[0,:],mod_y_x[1,:])]  \

    cv2.imshow('super',super)
  
    cv2.waitKey(1)
    