# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 11:01:31 2016

@author: sunil.bhandari1
""" 
def optimiz(covm,LongTermVol,weigthTuple,Gap,calDate,df_input_clean):
    
    import RegionalFactor
    import TrendIndcatorCal
    import scipy.optimize as opt
    import numpy as np
    import pandas as pd
    print("inside optimize")
         
    #objective Function 
    def ObjERC(t):               
        fval=np.multiply(np.sum((np.multiply(t,expReturn))),-1) 
        #print("At weight: ",t,"Func value is : ",fval)
        return fval
             
    #initial weights          
    t=np.ones((len(covm),1))
    #t = np.multiply(t,.25)
    
    #TrendIndicator
#    trendIndicator = np.random.rand(1,22)    result expected
    trendIndicator = TrendIndcatorCal.trndCalculator(calDate,df_input_clean)
    #LongTermVol
    # LongTermVol = np.array([.0329,.0774,.0186,.0283,.0492,.064,.032,.0574,.0577,.0758,.0908,.0573,.0719,.1514,.1808,.1694,.213,.2229,.1957,.2613,.2747,.2855])
    
    #RegFactor
    regFactor = RegionalFactor.regFactor(calDate)
    #regFactor = np.random.rand(1,22)
    #ExpecRet
    expReturn = trendIndicator*LongTermVol*regFactor
    
#    bounds = tuple([ (0.015385,.046154),(0.015385,.046154),(0.015385,.046154),(0.015385,.046154),(0.015385,.046154),(0.015385,.046154),(0.015385,.046154),(0.015385,.046154),(0.015385,.046154),(.023077,.069231),(.007692,.023077),(.015385,.046154),(.015385,.046154),(.167373,.502118),(.073314,.219942),(.024147,.072441),(.013729,.041188),(.005269,.015807),(.00563,.016891),(.005118,.015355),(.002619,.007858),(.0028,.0084)]) 
#    Gap = np.array([.025,.1,.025,.05,.1,.143,.143,.167,.167,.2,.2,.167,.167,.2,.2,.2,.2,.25,.25,.25,.25,.25])
    bounds = weigthTuple
    cons = ({'type': 'ineq','fun' : lambda t: .1 - np.sqrt(np.dot(np.dot(np.transpose(t),covm),t)) },
            {'type': 'ineq','fun' : lambda t: .2 - np.sum(np.multiply(np.transpose(t),Gap))},
            {'type': 'ineq','fun' : lambda t: 1 - np.sum(t)},
            {'type': 'ineq','fun' : lambda t: np.sum(t)} 
             )
     
    io=opt.minimize(ObjERC,t,bounds=bounds,constraints=cons)
    return io


