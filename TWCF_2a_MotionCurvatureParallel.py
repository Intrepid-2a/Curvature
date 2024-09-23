"""
Motion curvature estimation across blind spot
TWCF IIT vs PP experiment 2a piloting
Authors: Belén María Montabes de la Cruz, Clement Abbatecola
    Code Version:
        2.0 # 2024/04/09    Final common version before eye tracking
        3.0 # 2024/03/07    Common version with Eye tracking version
"""

import psychopy
from psychopy import core, visual, gui, data, event
from psychopy.tools.coordinatetools import pol2cart, cart2pol
from psychopy.tools.mathtools import distance
import numpy as np
from numpy import ndarray
import random, datetime, sys, os
import math
from math import sin, cos, radians, pi
from glob import glob
from itertools import compress
#from curvature import placeCurvatureDots
sys.path.append(os.path.join('..', 'EyeTracking'))
from EyeTracking import localizeSetup, EyeTracker, fusionStim

###### Curvature function
def placeCurvatureDots(B, C, curvature):

    print([B, C, curvature])

    # B: coordinates of point B
    # C: coordinates of point C
    # curvature: the amount of curvature for the previous and next points

    # Assuming the time that passes between presentation of B and C
    # is equal to the time passing between A and B and between C and D
    # the location of A and D can be calculated, such that the 2 triplets 
    # of points have the specified curvature.

    # First, we need B to be lower on the screen than C:
    if B[1] > C[1]:
        B, C = C, B

    # Then, if the specified curvature is 0, this is a special case
    # for which the equation doesn't work...
    #
    # ... but the 4 points should lie on a straight line:

    if curvature == 0:
        A = [B[0] - (C[0]-B[0]), B[1] - (C[1]-B[1])]
        D = [C[0] + (C[0]-B[0]), C[1] + (C[1]-B[1])]
        # we return this result:
        # return([A, B, C, D]) # this returned a tuple, while for non-zero curvature, the function returns an array
        return(np.array([A,B,C,D])[0,:,:])

    # If the curvature is not 0, we need to do some more work.

    # distance between B and C:
    dist = ((B[0] - C[0])**2 + (B[1] - C[1])**2)**0.5
    
    # print(dist)

    # the radius of the circle describing the curvature:
    R = 1 / np.abs(curvature)

    # print(R)

    # The angle between two lines drawn through the origin
    # of the circle of curvature and the two points:
    ang_rad = 2 * ( (np.pi/2) - np.arccos( (dist/2) / R ) )

    # print(ang_rad)

    # Get the angle in radians for all 4 points,
    # with B and C in the middle:
    point_angles = [ang_rad * x for x in [-1.5,-.5,.5,1.5]]
    
    # print(point_angles)

    # Now get the coordinates of the 4 points:
    # point_coords = [[np.cos(xa)*R, np.sin(xa)*R] for xa in point_angles]
    # in an array:
    point_coords = np.array([np.cos(point_angles)*R, np.sin(point_angles)*R]).T

    # print(point_coords)
    
    # Right now, the curvature is always toward fixation
    # but the relative placement is correct,
    # we just need to move things around a bit.

    # First we correct for our positive and negative curvature.
    # This does not really exist, we just define negative curvature
    # to mean 'away from fixation'.

    point_coords = point_coords - [R,0]
    if curvature < 0:
        point_coords[:,0] *= -1
    
    # Now we flip the points if the original B and C are to the left:
    if np.mean([B[0], C[0]]) < 0:
        point_coords[:,0] *= -1
    
    # Then we reposition the points such that the current B (2nd) point
    # is at [0,0]:
    point_coords -= point_coords[1,:]

    # We get the original and current angle of a line through the 2nd
    # and 3rd point:
    orig_ang = np.arctan2(C[1]-B[1], C[0]-B[0])
    curr_ang = np.arctan2(point_coords[2,1], point_coords[2,0])

    # Rotate the current points by the difference to match the input orientation:
    th = orig_ang - curr_ang
    Rm = np.array([[np.cos(th), -1*np.sin(th)],[np.sin(th),np.cos(th)]])
    point_coords = Rm @ point_coords.T

    # print(point_coords)

    # Translate such that the second and third point match the input locations:
    point_coords = point_coords.T + B

    # That should be all, so we return all 4 coordinates:
    return(point_coords)


######
#### Initialize experiment
######

def doCurvatureTask(ID=None, hem=None, location=None):
    ## parameters
    Nrevs   = 10   #
    Ntrials = 30  # at least 10 reversals and 30 trials for each staircase (~ 30*8 staircases = 250 trials)
    letter_height = 1
    
    # site specific handling
    if location == None:
        if os.sys.platform == 'linux':
            location = 'toronto'
        else:
            location = 'glasgow'

    if location == 'glasgow':
        
        ## path
        expInfo = {'ID':'test', 'hemifield':['left','right']}
        dlg = gui.DlgFromDict(expInfo, title='Infos', screen=0)
        ID = expInfo['ID'].lower()
        hem = expInfo['hemifield']

        main_path = '../data/curvature/'
        data_path = main_path
        eyetracking_path = main_path + 'eyetracking/' + ID + '/'
        
        os.makedirs(data_path, exist_ok=True)
        os.makedirs(eyetracking_path, exist_ok=True)
        
        x=1
        filename = ID + '_Curvature_' + ('LH' if hem == 'left' else 'RH') + '_'
        while (filename + str(x) + '.txt') in os.listdir(data_path):
            x += 1
        y=1
        et_filename = ID + 'Curvature' + ('LH' if hem == 'left' else 'RH') + '_'
        while len(glob(eyetracking_path + et_filename + str(y) + '.*')):
            y += 1

        ## blindspot
        bs_file = open(glob(main_path + 'mapping/' + ID + '_LH_blindspot*.txt')[-1], 'r')
        bs_param = bs_file.read().replace('\t','\n').split('\n')
        bs_file.close()
        spot_left_cart = eval(bs_param[1])
        spot_left = cart2pol(spot_left_cart[0], spot_left_cart[1])
        spot_left_size = eval(bs_param[3])
    
        bs_file = open(glob(main_path + 'mapping/' + ID + '_RH_blindspot*.txt')[-1],'r')
        bs_param = bs_file.read().replace('\t','\n').split('\n')
        bs_file.close()
        spot_righ_cart = eval(bs_param[1])
        spot_righ = cart2pol(spot_righ_cart[0], spot_righ_cart[1])
        spot_righ_size = eval(bs_param[3])

        if hem == 'left':
            spot_cart = spot_left_cart
            spot      = spot_left
            spot_size = spot_left_size
        else:
            spot_cart = spot_righ_cart
            spot      = spot_righ
            spot_size = spot_righ_size

        # Padding angles =  BSheight/3 + 2 (dotwidth) + 1(padding) --> value obtained from piloting
        ## bs_prop = setup['blindspotmarkers'][hemifield+'_prop'] --what is prop? We don't get that
        ## angpad = (bs_prop['size'][1]/3) + 2 + 1 
        angpad = spot_size[1]/3 + 2 + 1

        # Padding on circle side
        ## side = (bs_prop['size'][1] - bs_prop['size'][0]) * 0.15/0.5
        side = (spot_size[1] - spot_size[0])*0.15/0.5
    
        ## colours
        col_file = open(glob(main_path + 'color/' + ID + '_col_cal*.txt')[-1],'r')
        col_param = col_file.read().replace('\t','\n').split('\n')
        col_file.close()
        col_ipsi = eval(col_param[3]) if hem == 'left' else eval(col_param[5]) # left or right
        col_cont = eval(col_param[5]) if hem == 'left' else eval(col_param[3]) # right or left
        col_back   = [ 0.55, 0.45,  -1.0]  #changed by belen to prevent red bleed
        col_both = [eval(col_param[3])[1], eval(col_param[5])[0], -1] 
    
        ## window & elements
        win = visual.Window([1500,800],allowGUI=True, monitor='ExpMon',screen=1, units='deg', viewPos = [0,0], fullscr = True, color= col_back)
        win.mouseVisible = False
        fixation = visual.ShapeStim(win, vertices = ((0, -2), (0, 2), (0,0), (-2, 0), (2, 0)), lineWidth = 4, units = 'pix', size = (10, 10), closeShape = False, lineColor = col_both)
        xfix = visual.ShapeStim(win, vertices = ((-2, -2), (2, 2), (0,0), (-2, 2), (2, -2)), lineWidth = 4, units = 'pix', size = (10, 10), closeShape = False, lineColor = col_both)

        ## Fusion Stimuli

        hiFusion = fusionStim(win = win, pos = [0, 7], colors = [col_both,col_back])
        loFusion = fusionStim(win = win, pos = [0,-7], colors = [col_both,col_back]) 

        ## BS stimuli
        blindspot = visual.Circle(win, radius = .5, pos = [7,0], units = 'deg', fillColor=col_ipsi, lineColor = None)
        blindspot.pos = spot_cart
        blindspot.size = spot_size
        
        ## eyetracking
        colors = {'both'   : col_both,
                  'back'   : col_back} 
        tracker = EyeTracker(tracker           = 'mouse',
                             trackEyes         = [True, True],
                             fixationWindow    = 2.0,
                             minFixDur         = 0.2,
                             fixTimeout        = 3.0,
                             psychopyWindow    = win,
                             filefolder        = eyetracking_path,
                             filename          = et_filename+str(y),
                             samplemode        = 'average',
                             calibrationpoints = 5,
                             colors            = colors)

    elif location == 'toronto':
    
        # not sure what you want to do here, maybe check if parameters are defined, otherwise throw an error? Or keep the gui in that case?
        
        
        expInfo = {}
        askQuestions = False
        if ID == None:
            expInfo['ID'] = ''
            askQuestions = True
        if hem == None:
            expInfo['hemifield'] = ['left','right']
            askQuestions = True
        if askQuestions:
            dlg = gui.DlgFromDict(expInfo, title='Infos', screen=0)

        if ID == None:
            ID = expInfo['ID'].lower()
        if hem == None:
            hem = expInfo['hemifield']
        
        ## path
        main_path = '../data/curvature/'
        data_path = main_path
        eyetracking_path = main_path + 'eyetracking/' + ID + '/'
        x = 1
        filename = ID + '_dist_' + ('LH' if hem == 'left' else 'RH') + '_'
        while (filename + str(x) + '.txt') in os.listdir(data_path):
            x += 1
        y = 1
        et_filename = ID + '_dist_' + ('LH' if hem == 'left' else 'RH') + '_'
        while len(glob(eyetracking_path + et_filename + str(y) + '.*')):
            y += 1
        
        # this _should_ already be handled by the Runner utility: setupDataFolders()
        os.makedirs(data_path, exist_ok=True)
        os.makedirs(eyetracking_path, exist_ok=True)
        
        
        trackEyes = [True, True]
        
        # get everything shared from central:
        setup = localizeSetup(location=location, trackEyes=trackEyes, filefolder=eyetracking_path, filename=et_filename+str(y), task='distance', ID=ID) # data path is for the mapping data, not the eye-tracker data!
    
        # unpack all this
        win = setup['win']
    
        colors = setup['colors']
        col_both = colors['both']
        if hem == 'left':
            col_ipsi, col_contra = colors['left'], colors['right']
        if hem == 'right':
            col_contra, col_ipsi = colors['left'], colors['right']
    
        hiFusion = setup['fusion']['hi']
        loFusion = setup['fusion']['lo']
    
        blindspot = setup['blindspotmarkers'][hem]
        
        fixation = setup['fixation']
    
        tracker = setup['tracker']
        
    else:
        raise ValueError("Location should be 'glasgow' or 'toronto', was {}".format(location))

    # create output files:
    respFile = open(data_path + filename + str(x) + '.txt','w')
    respFile.write(''.join(map(str, ['Start: \t' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') + '\n'])))
    respFile.write('\t'.join(map(str, ['TrialN',
                                    'Curvature',
                                    'Stimulus_position', 
                                    'GreenStim', 
                                    'Staircase', 
                                    'Response', 
                                    'Reversal', 
                                    'AllTrials', 
                                    'StairsOngoing'])) + '\n')
    respFile.close()
    gazeFile = open(eyetracking_path + filename + str(x) + '_gaze.txt','w')
    gazeFile.write("Trial\tPosition\tEye_pos\tStaircase\tTime\tGaze\n")
    gazeFile.close()


    

    ######
    ## Prepare stimulation
    ######


    ## stimuli
    point1 = visual.Circle(win, radius = .7, pos = pol2cart(00, 6), fillColor = 'white', lineColor = None, units = 'deg')
    point2 = visual.Circle(win, radius = .7, pos = pol2cart(00, 6), fillColor = 'white', lineColor = None, units = 'deg')
    point3 = visual.Circle(win, radius = .7, pos = pol2cart(00, 6), fillColor = 'white', lineColor = None, units = 'deg')
    point4 = visual.Circle(win, radius = .7, pos = pol2cart(00, 6), fillColor = 'white', lineColor = None, units = 'deg')

    # Additional variables 

    # distance between dots in the trajectory should be related to the height of the blind spot:
    dot_distance = 2

    # stimulus width should be related to dot offset at maximum curvature:
    max_curvature_points = placeCurvatureDots(  B = [0,dot_distance/2],
                                                C = [0,-dot_distance/2],
                                                curvature = 0.4)
    stim_width = np.abs(max_curvature_points[0,0]) + 0.7 #dividing 0.7 by 2 (half size to the edge of the stimulus)

    #arc_length = (spot_size[1] * 1.0) 

    # and with a radius that moves away from the blind spot center, toward fixation with enough padding (stim width and 2 dva extra)
    r = np.sqrt( (abs(spot_cart[0]) - (spot_size[0]/2) -0.5-stim_width)**2 + spot_cart[1]**2 )#(abs(spot_cart[0]) - (spot_size[0]/2) - 2 - stim_width)**2 + spot_cart[1]**2 )

    # the direction by which the 'above' blind spot position is rotated, depends on hem:
    ang_mod = 1 if hem == 'right' else -1

    # at blind spot middle of trajectory:
    if hem == 'right':
        bsm_x = spot_cart[0] - (spot_size[0]/2) -0.5- stim_width
        instructions = visual.TextStim(win, text="Throughout the experiment you will fixate at a a cross located at the centre of the screen. It is important that you maintain fixation on this cross at all times.\n\n In every trial you will be presented with a dot which will move along a curve. You will have to indicate with a keypress if the dot's motion was curved towards fixation or away from fixation  \n \nLeft arrow = motion curved towards fixation.\n \n Right arrow = motion curved away from fixation.\n\n\n You will only be able to respond when the fixation cross rotates from a '+' to a 'x' \n\n\n Press the space bar when you're ready to start the experiment.", color=col_both)

    else:
        bsm_x = spot_cart[0] + (spot_size[0]/2) +0.5+ stim_width
        instructions = visual.TextStim(win, text="Throughout the experiment you will fixate at a a cross located at the centre of the screen. It is important that you maintain fixation on this cross at all times.\n\n In every trial you will be presented with a dot which will move along a curve. You will have to indicate with a keypress if the dot's motion was curved towards fixation or away from fixation  \n \nLeft arrow = motion curved away from fixation.\n \n Right arrow = motion curved towards fixation.\n\n\nYou will only be able to respond when the fixation cross rotates from a '+' to a 'x' \n\n\n Press the space bar when you're ready to start the experiment.", color = col_both)

    bsm = [bsm_x, spot_cart[1]]

    # convert angle to slope:
    slope = 0
    intercept = spot_cart[1] + spot_size[1] + 1.4

    # this is using a diagonal of sorts
    A = (slope**2 + 1)
    B = 2*((slope*intercept)) # the other parts are 0 since there is no offset from the origin
    C = (-1*(r**2) + intercept**2) # again the parts containing p and q are zero

    # now we can use the quadratic formula:
    #x1 = (-1*B) + np.sqrt(B**2 - (4*A*C)) / 2*A
    #x2 = (-1*B) - np.sqrt(B**2 - (4*A*C)) / 2*A

    #x = max([x1, x2])

    ang_up = ( np.arctan( ((spot_size[1]/2) + 0.7 + 1) / r ) / np.pi ) * 180 * 2


    if hem == 'right':
        positions = [ [ [sum(x) for x in zip(pol2cart(cart2pol(bsm[0], bsm[1])[0] - (ang_up*ang_mod), r), [0,dot_distance/2 ])],
                        [sum(x) for x in zip(pol2cart(cart2pol(bsm[0], bsm[1])[0] - (ang_up*ang_mod), r), [0,dot_distance/-2])] ],
                    [ [bsm[0], bsm[1]-(dot_distance/2)],
                        [bsm[0], bsm[1]+(dot_distance/2)] ] ]
    else:
        positions = [ [ [sum(x) for x in zip(pol2cart(cart2pol(bsm[0], bsm[1])[0] + (ang_up*ang_mod), r), [0,dot_distance/2 ])],
                        [sum(x) for x in zip(pol2cart(cart2pol(bsm[0], bsm[1])[0] + (ang_up*ang_mod), r), [0,dot_distance/-2])] ],
                    [ [bsm[0], bsm[1]+(dot_distance/2)],
                        [bsm[0], bsm[1]-(dot_distance/2)] ] ]
    


    instructions.wrapWidth = 30
    instructions.draw()
    win.flip()
    event.waitKeys(keyList='space')

    ######
    #### Prepare eye tracking
    ######

  
    


    ## setup and initialize eye-tracker
    blindspot.autoDraw = True
    fixation.draw()
    win.flip()

    if tracker.tracker == 'eyelink':
        tracker.initialize(calibrationScale=(0.35, 0.35))
    else:
        tracker.initialize()
    tracker.calibrate()


    k = event.waitKeys()
    if k[0] in ['q']:
        win.close()
        core.quit()

    tracker.startcollecting()


    ######
    #### Staircase
    ######
    
    ## Curvatures, note that 0.000001 instead of 0 to avoid crushing
    #curvature = [0.4,  0.35,  0.3, 0.25, 0.2,  0.15,  0.1,  0.05,  0.000001,-0.000001, -0.05,  -0.1,  -0.15,  -0.2, -0.25,  -0.3,  -0.35,  -0.4]
    curvature = [round((x / 20)-0.4, ndigits=3) for x in list(range(0,17))]   # NEW 17 points only

    ##Staircase parameters 
    step = [[[0, 0], [0, 0]], [[0, 0], [0, 0]]] #[['left', 'right'], ['left', 'right']]
    direction = [[[1, -1], [1, -1]], [[1, -1], [1, -1]]] # 2 directions per eye and position converging to straight
    whicheye = ['left', 'right'] 
    revs = [[0, 0], [0, 0]], [[0, 0], [0, 0]] #counter for the number of reversals 
    trial = [[0, 0], [0, 0]], [[0, 0], [0, 0]] #counter for the trail number
    resps = [[[[], []], [[], []]], [[[], []], [[], []]]] #keeps track of responses for the reversals
    stairs_ongoing = [[[True, True], [True, True]], [[True, True], [True, True]]] #to end experiment [left pos0, right pos0] [left pos1, right pos1] 
    not_ongoing = [[[False, False], [False, False]], [[False, False], [False, False]]] #to end experiment  
    abort = False
    recalibrate = False
    choice = []

    #keeping track of time
    trial_clock = core.Clock()

    #repeated draws  
    def repeat_draw():
        fixation.draw()
        hiFusion.draw()
        loFusion.draw()
    

    while not stairs_ongoing == not_ongoing:
        increment = True
        #1. Select the position to draw on
        if  stairs_ongoing[0] == [[False, False], [False, False]]: # doing all(stairs_ongoing[1]) leads to error = 'list indices must be integers or slices, not list' 
            position = 1
            #print('stair 0 = ', stairs_ongoing[0],'pos =', position)
        elif stairs_ongoing[1] == [[False, False], [False, False]]:#if position1=done   --- try any
            position = 0
            #print('stair 0 = ', stairs_ongoing[1],'pos =', position)
        elif any(stairs_ongoing[0]) == True and any(stairs_ongoing[1]) == True:
            position = np.random.choice([0, 1]) # 1 = above, 2 = BS
            #print('both stairs ongoing, pos = ', position) 
        #2. Select the eye to stimulate
        if stairs_ongoing[position][0] == [False, False]:
            eye = 1
        elif stairs_ongoing[position][1] == [False, False]:
            eye = 0
        elif any(stairs_ongoing[position][0]) == True and any(stairs_ongoing[position][1]) == True:
            eye = np.random.choice([0, 1])
        # 3. Select the staircase to draw (i.e. curvature towards or curvature away)
        staircase = np.random.choice(list(compress([0, 1], stairs_ongoing[position][eye])))
        fixation.color = col_both 

        ##position of central dots (fixed, either above or around BS)
        point2.pos = positions[position][0]
        point3.pos = positions[position][1]
        
        ##position of first and fourth dots (mobile, either curved towards or curved away)
        tstep = step[position][eye][staircase] if step[position][eye][staircase] >0 else step[position][eye][staircase]*-1 #-1 to prevent it from being negative
        currentcurv = direction[position][eye][staircase] * curvature[tstep]
        print('currently we are at', currentcurv, 'current step =', tstep)
        coords = placeCurvatureDots(point2.pos, point3.pos, currentcurv)
        point1.pos = coords[3]
        point4.pos = coords[0]

        #color of dots - which eye to stimulate 
        if whicheye[eye] == hem:
            point1.fillColor = col_ipsi
            point2.fillColor = col_ipsi
            point3.fillColor = col_ipsi
            point4.fillColor = col_ipsi
        else:
            point1.fillColor = col_cont
            point2.fillColor = col_cont
            point3.fillColor = col_cont
            point4.fillColor = col_cont

        #resetting fusion stimuli
        hiFusion.resetProperties()
        loFusion.resetProperties()

        gaze_out = False

        ## pre trial fixation 
        tracker.comment('pre-fixation')

        if not tracker.waitForFixation(fixationStimuli = [fixation, hiFusion, loFusion]):
            recalibrate = True
            gaze_out = True


        ## commencing trial 
        if not gaze_out:
            ## trial

            stim_comments = ['start trace, pos=1', 'pos = 2', 'pos = 3', 'pos = 4', 'return, pos = 4',
                'pos = 3', 'pos = 2', 'pos = 1', 'final go, pos = 1', 'pos = 2', 'pos = 3', 'end trace, pos = 4', 'trace = off'] #BM what's this?
            tracker.comment('start trial %d'%(trial[position][eye][staircase]))
            
            
            gazeFile = open(eyetracking_path + filename + str(x) + '_gaze.txt','a')
            trial_clock.reset()
            while trial_clock.getTime() < 1.4 and not abort:
                t = trial_clock.getTime()
                
                gazeFile.write('\t'.join(map(str, [trial[position][eye][staircase],
                               position,
                               eye,
                               staircase,
                               round(t,2),
                               tracker.lastsample()])) + "\n")
                
                if not tracker.gazeInFixationWindow():
                    gaze_out = True
                    finaldiff = 'Trial aborted'
                    print('gaze out')
                    break

                repeat_draw()

                #drawing the stimuli
                
               

                if .1 <= trial_clock.getTime() < .2:
                    if len(stim_comments) == 13:
                        tracker.comment(stim_comments.pop()) 
                    repeat_draw()
                    point1.draw()
                    print('point1')
                elif .2 <= trial_clock.getTime() < .3:
                    if len(stim_comments) == 12:
                        tracker.comment(stim_comments.pop()) 
                    repeat_draw()
                    point2.draw()
                elif .3 <=  trial_clock.getTime() < .4:
                    if len(stim_comments) == 11:
                        tracker.comment(stim_comments.pop()) 
                    repeat_draw()
                    point3.draw()
                elif .4 <= trial_clock.getTime() < .5:
                    if len(stim_comments) == 10:
                        tracker.comment(stim_comments.pop()) 
                    repeat_draw()
                    point4.draw()
                elif .5 <= trial_clock.getTime() < .6:
                    if len(stim_comments) == 9:
                        tracker.comment(stim_comments.pop()) 
                    repeat_draw()
                    point4.draw()
                elif .6 <= trial_clock.getTime() < .7:
                    if len(stim_comments) == 8:
                        tracker.comment(stim_comments.pop()) 
                    repeat_draw()
                    point3.draw()
                elif .7 <= trial_clock.getTime() < .8:
                    if len(stim_comments) == 8:
                        tracker.comment(stim_comments.pop()) 
                    repeat_draw()
                    point2.draw()
                elif .8 <= trial_clock.getTime() < .9:
                    if len(stim_comments) == 6:
                        tracker.comment(stim_comments.pop()) 
                    repeat_draw()
                    point1.draw()
                elif .9 <= trial_clock.getTime() < 1.0:
                    if len(stim_comments) == 5:
                        tracker.comment(stim_comments.pop()) 
                    repeat_draw()
                    point1.draw()
                elif 1.0 <= trial_clock.getTime() < 1.1:
                    if len(stim_comments) == 4:
                        tracker.comment(stim_comments.pop()) 
                    repeat_draw()
                    point2.draw()
                elif 1.1 <=  trial_clock.getTime() < 1.2:
                    if len(stim_comments) == 3:
                        tracker.comment(stim_comments.pop()) 
                    repeat_draw()
                    point3.draw()
                elif 1.2 <= trial_clock.getTime() < 1.3:
                    if len(stim_comments) == 2:
                        tracker.comment(stim_comments.pop()) 
                    repeat_draw()
                    point4.draw()
                

                win.flip()


                k = event.getKeys(['q'])
                if k and 'q' in k:
                    abort = True
                    break
                
            if len(stim_comments) == 1:
                tracker.comment(stim_comments.pop()) # end stimulation
            gazeFile.close()

                
        if abort:
            break

        if not gaze_out:
        #Wait for responses
            hiFusion.draw()
            loFusion.draw()
            xfix.draw()
            win.flip()

            k = ['wait']
            while k[0] not in ['q', 'space', 'left', 'right']:
                k = event.waitKeys()
            if hem == 'right':
                if k[0] in ['q']:
                    abort = True
                    break
                elif k[0] in ['left']:
                    if currentcurv == .4:
                        pass
                    else:
                        if step[position][eye][staircase] <= len(curvature)-2:
                            step[position][eye][staircase] += 1
                            print(step[position][eye][staircase])
                            choice = 'left'
                        if step[position][eye][staircase] ==len(curvature)-1:
                            step[position][eye][staircase] -= 1
                            print(step[position][eye][staircase])
                            choice= 'NA'
                    trial_clock.reset()
                elif k[0] in ['right']:
                    if currentcurv == -.4:
                        pass
                    else:
                        if step[position][eye][staircase] <= len(curvature)-2:
                            step[position][eye][staircase] -= 1
                            choice = 'right'
                            print(step[position][eye][staircase])
                        if step[position][eye][staircase] ==len(curvature)-1:
                            step[position][eye][staircase] += 1
                            print(step[position][eye][staircase])
                            choice= 'NA'
                    trial_clock.reset()
                elif k[0] in ['space']:
                    choice = 'Trial aborted'
                    increment = False
                    trial_clock.reset()
            else: ## add -4 +4
                if k[0] in ['q']:
                    abort = True
                    break
                elif k[0] in ['right']:
                    if currentcurv == .4:
                        pass
                    else:
                        if step[position][eye][staircase] <= len(curvature)-2:
                            step[position][eye][staircase] += 1
                            choice = 'right'
                            print(step[position][eye][staircase])
                        if step[position][eye][staircase] ==len(curvature)-1:
                            step[position][eye][staircase] -= 1
                            print(step[position][eye][staircase])
                            choice= 'NA'
                    trial_clock.reset()
                elif k[0] in ['left']:
                    if currentcurv == -.4:
                        pass
                    else:
                        if step[position][eye][staircase] <= len(curvature)-2:
                            step[position][eye][staircase] -= 1
                            print(step[position][eye][staircase])
                            choice = 'left'
                        if step[position][eye][staircase] ==len(curvature)-1:
                            step[position][eye][staircase] += 1
                            print(step[position][eye][staircase])
                            choice= 'NA'
                    trial_clock.reset()
                elif k[0] in ['space']:
                    choice = 'Trial aborted'
                    trial_clock.reset()
        else:
            if recalibrate:
                recalibrate = False
                tracker.calibrate()
                win.flip()
                fixation.draw()
                win.flip()
                k = event.waitKeys()
                if k[0] in ['q']:
                    abort = True
                    break
            # changing fixation to signify gaze out, restart with 'up' possibily of break and manual recalibration 'r' 
            else:
                hiFusion.draw()
                loFusion.draw()
                visual.TextStim(win, '#', height = letter_height, color = col_both).draw()
                win.flip()
                k = ['wait']
                while k[0] not in ['q', 'up', 'r']:
                    k = event.waitKeys()
                if k[0] in ['q']:
                    abort = True
                    break
        
                # manual recalibrate
                if k[0] in ['r']:
                    tracker.calibrate()
                    win.flip()
                    fixation.draw()
                    win.flip()
                    k = event.waitKeys()
                    if k[0] in ['q']:
                        abort = True
                        break
            increment = False
            ##Adapting the staircase 
            if choice in ['left', 'right']:
                resps[position][eye][staircase] = resps[position][eye][staircase] + [choice]


        if increment:
        #sets the bounds for the staircase 
           ## Reversals
            resps[position][eye][staircase]  = resps[position][eye][staircase]  + [choice]
            if resps[position][eye][staircase][-2:] == ['left', 'right'] or resps[position][eye][staircase][-2:] == ['right', 'left']: 
                revs[position][eye][staircase]  = revs[position][eye][staircase]  + 1


        #writing reponse file
        respFile = open(data_path + filename + str(x) + '.txt','a')
        respFile.write('\t'.join(map(str, [trial[position][eye][staircase], 
                                        currentcurv,# Stimulus location
                                        position,
                                        eye, 
                                        staircase,
                                        choice, 
                                        revs,
                                        trial,
                                        stairs_ongoing])) + "\n") #block
        respFile.close()
        #final updates
        if not choice == 'Trial aborted':
            trial[position][eye][staircase]  = trial[position][eye][staircase]  + 1
        else:
            pass
        ##Check if experiment can continue
        stairs_ongoing[position][eye][staircase]  = revs[position][eye][staircase]  <= Nrevs or trial[position][eye][staircase]  < Ntrials

    ## Closing prints
    if abort:
        respFile = open(data_path + filename + str(x) + '.txt','a')
        respFile.write("Run manually ended at " + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + "!")
        respFile.close()
        bye = visual.TextStim(win, text="Run manually ended \n Press space bar to exit")
    elif stairs_ongoing == not_ongoing:
        print('run ended properly!')
        bye = visual.TextStim(win, text="You have now completed the experimental run. Thank you for your participation!! \n Press space bar to exit")
    else:
        respFile = open(data_path + filename + str(x) + '.txt','a')
        respFile.write("something weird happened")
        respFile.close()
        print('something weird happened')

    print(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
    blindspot.autoDraw = False

    ## Farewells
    bye.draw()
    win.flip()
    core.wait(3)
    
    tracker.shutdown()
    win.close()
    core.quit()

if __name__ == "__main__": 
    doCurvatureTask()
