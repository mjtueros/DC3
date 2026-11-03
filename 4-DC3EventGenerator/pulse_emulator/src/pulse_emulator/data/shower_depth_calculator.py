import numpy as np

kRearth = 6370949.

def ComputeGrammage(Zenith_, XmaxDistance_, ShowerCoreHeight_, InjectionHeight_, LongitudinalDistance_):
    if np.isscalar(Zenith_):
        _grammage = ComputeDistanceGrammage(Zenith_, XmaxDistance_, LongitudinalDistance_, ShowerCoreHeight_)
    else:
        _grammage = [ComputeDistanceGrammage(Zenith_[i], XmaxDistance_[i], LongitudinalDistance_[i], ShowerCoreHeight_[i]) for i in range(len(Zenith_))]
    return np.array(_grammage)


def ComputeInjectionPoint(Azimuth_, Zenith_, InjectionHeight_, ShowerCoreHeight_):
    k_shower = np.array([np.cos(Azimuth_*np.pi/180.)*np.sin(Zenith_*np.pi/180),np.sin(Azimuth_*np.pi/180.)*np.sin(Zenith_*np.pi/180), np.cos(Zenith_*np.pi/180)])
    _delta = (kRearth + ShowerCoreHeight_)**2*np.cos(Zenith_*np.pi/180.)**2 + (InjectionHeight_ - ShowerCoreHeight_)*(InjectionHeight_ + ShowerCoreHeight_ + 2.*kRearth)
    _injection_length = (kRearth + ShowerCoreHeight_)*np.cos(Zenith_*np.pi/180.) + np.sqrt(_delta)
    InjectionX = - k_shower[0]*_injection_length
    InjectionY = - k_shower[1]*_injection_length
    InjectionZ = - k_shower[2]*_injection_length + ShowerCoreHeight_

    return np.array([InjectionX, InjectionY, InjectionZ])

def ComputeLongitudinalDistance(Azimuth_, Zenith_, InjectionHeight_, ShowerCoreHeight_, XSourceRec_, YSourceRec_, ZSourceRec_):
    if np.isscalar(Zenith_):
        _L = np.linalg.norm(np.array([XSourceRec_, YSourceRec_, ZSourceRec_]) - ComputeInjectionPoint(Azimuth_, Zenith_, InjectionHeight_, ShowerCoreHeight_))
    else:
        _L = [np.linalg.norm(np.array([XSourceRec_[i], YSourceRec_[i], ZSourceRec_[i]]) - ComputeInjectionPoint(Azimuth_[i], Zenith_[i], InjectionHeight_[i], ShowerCoreHeight_[i])) for i in range(len(Zenith_))]

    return _L

def GetLocalZenith(Zenith_, LocalHeight_, StartHeight_):
    '''
    Compute zenith angle at any point along the erath curvature
    Inputs: Zenith_, InjectionHeight_, ShowerCoreHeight_
    Outputs: Zenith angle at given location
    '''
    _delta = (kRearth + StartHeight_)**2*np.cos(Zenith_*np.pi/180.)**2 + (LocalHeight_ - StartHeight_)*(LocalHeight_ + StartHeight_ + 2.*kRearth)
    _path_length = (kRearth + StartHeight_)*np.cos(Zenith_*np.pi/180.) + np.sqrt(_delta)
    _Zenith_at = (np.pi-np.arccos((_path_length**2 + (kRearth + LocalHeight_)**2 - (kRearth + StartHeight_)**2)/(2.*_path_length*(kRearth + LocalHeight_))))*180./np.pi

    return _Zenith_at

def GetLocalHeight(Zenith_, StartHeight_, PathLength_):
    _height_at =  -kRearth + np.sqrt((kRearth + StartHeight_)**2 + PathLength_**2 - 2.*PathLength_*(kRearth+StartHeight_)*np.cos(Zenith_*np.pi/180.))
    return _height_at

def GetDensity(_height,model):
    if model == "isothermal":
        #Using isothermal Model
        rho_0 = 1.225    #kg/m^3
        M = 0.028966    #kg/mol
        g = 9.81        #m.s^-2
        T = 288.
        R = 8.32        #J/K/mol , J=kg m2/s2
        rho = rho_0*np.exp(-g*M*_height/(R*T))  # kg/m3

    elif model == "linsley":
        
        # Fitted values
        bl = np.array([1183.356719, 1118.314131, 1144.771295, 1162.244263, 373.099992, 0.967112])*10
        cl = -1/np.array([-9.48131e-05, -0.0001050906, -0.0001434887, -0.0001447813, -0.0001505947, -1.0e-7])
        hl = np.array([4.3008, 9.0446, 27.3293, 95.0855, 242.1879, 426.6054])*1e3


        if _height>=hl[-1]:  # no more air
            rho = 0
        else:
            hlinf = np.array([0] + list(hl[:-1]))  #m
            ind = np.logical_and([_height>=hlinf],[_height<hl])[0]
            rho = bl[ind]/cl[ind]*np.exp(-_height/cl[ind])
            #print(rho, ind, _height)
            rho = rho[0]
    else:
        print("#### Error in GetDensity: model can only be isothermal or linsley.")
        return 0

    return rho

def ComputeDistanceGrammage(Zenith_, XmaxDistance_, LongitudinalDistance_, ShowerCoreHeight_):
    X = 0.
    dl_tot = 0
    conversion_factor = 0.1 #-> kg/m^-2 -> g/cm^-2

    _height = GetLocalHeight(Zenith_, ShowerCoreHeight_, XmaxDistance_)
    _zenith = GetLocalZenith(Zenith_, _height, ShowerCoreHeight_)
    nbe_iteration = 100
    dl = LongitudinalDistance_/nbe_iteration                  # 100 steps because no time for more
    #compute zenith at Xmax

    for dl in np.repeat(dl, nbe_iteration+1):                  #Do not start at 0...

        _height_new = GetLocalHeight(_zenith, _height, dl)
        _zenith_new = GetLocalZenith(_zenith, _height_new, _height)
        if _height_new <0: continue
        dX =  GetDensity(_height,'linsley')* dl * conversion_factor
        X += dX
        if np.isnan(X): print(LongitudinalDistance_, dl_tot, _height, _zenith, X, dX, Zenith_, ShowerCoreHeight_)
        #print(LongitudinalDistance_, dl_tot, _height_new, _zenith_new, X, dX, Zenith_, ShowerCoreHeight_)
        _height=_height_new
        _zenith = _zenith_new
        dl_tot +=dl

    return X
