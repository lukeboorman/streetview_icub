import matplotlib
#matplotlib.use('Agg')
import numpy as np
#from location import Location
from locationLuke import Location
from cffun import cf_beta
import DGStateAlan
import matplotlib.pyplot as plt

#TODO return list of actual and hat locations

def plotResults(path, hist, dictGrids, b_useNewDG, learningRate, note=""):

    str_title = hist.str_title +  "\nDG: %s learningRate: %f" % (b_useNewDG, learningRate ) 
    plt.figure()
    T=len(hist.ca1s)

    xys_bel  = np.zeros((T,2))
    ihds_bel = np.zeros((T,1))
    
    #N_places = (1+2*path.N_mazeSize)**2 # DO NOT USE
    
    #N_places = (1+2*dictGrids.Nmax)**2 # LB: Use this if needed

    for t in range(0,T):     
#        loc=Location()
        loc=Location(dictGrids) # Luke altered
#        loc.setGrids(hist.ca1s[t].grids, dictGrids)
        loc.setGrids(hist.ca1s[t].grids) # Luke

        xys_bel[t,:] = loc.getXY()
        ihds_bel[t]      = np.argmax(hist.ca1s[t].hd)

    #plt.subplot(4,2, (b_col2)+1)
    plt.subplot(4,1, 1)

    plt.plot(0.1+xys_bel[:,0], 'b')
    plt.hold(True)
    plt.plot(path.posLog[:,0], 'k')
    plt.ylabel('x location')

    plt.title(str_title)

#    plt.subplot(4,2, (b_col2)+3)
    plt.subplot(4,1, 2)

    plt.plot(0.1+xys_bel[:,1], 'b')
    plt.hold(True)
    plt.plot(path.posLog[:,1], 'k')
    plt.ylabel('y location')

    #head directions
#    plt.subplot(4,2, (b_col2)+5)
    plt.subplot(4,1, 3)
    plt.plot(path.posLog[:,2], 'k')   #ground truth
    plt.plot(ihds_bel, 'b')   #EC HD cells
    plt.ylabel('Heading')

#    plt.subplot(4,2, (b_col2)+7)
    plt.subplot(4,1, 4)


    plt.plot(hist.sub_errs, 'y')
    plt.hold(True)
    plt.plot(5*hist.sub_fires, 'b')
    plt.plot(hist.sub_ints, 'r')
    plt.ylim(0,1)

    plt.ylabel('Subiculum activation')
    plt.xlabel('time')

    b_losts = (np.sum(xys_bel!=path.posLog[:,0:2], 1)!=0)
    
    # COMMENTED OUT SAVING AS THE PATH IS BUGGY IN NOTEBOOK 
    #str = "Results/run_"+hist.str_title+"LR_%d_DG_%s_%s.eps" % (learningRate*1000, b_useNewDG, note)
    #plt.savefig(str)

    return (b_losts, xys_bel)


def plotErrors(hist1, hist2, hist3, lost1, lost2, lost3, learningRate, subTest = True, surfTest = True, placeTest=True, note=""):
    # Luke: adapated for newer plotErrorsN    
    hists=dict()
    losts=dict()
    hists[hist1.str_title]=hist1
    hists[hist2.str_title]=hist2        
    hists[hist3.str_title]=hist3
    losts[lost1.str_title]=lost1
    losts[lost2.str_title]=lost2        
    losts[lost3.str_title]=lost3    
    plotErrorsN(hists, losts, learningRate, subTest, surfTest, placeTest, note)
    
# Luke added N errors version! - takes in dictionay of info... lots more plotting available...
def plotErrorsN(hists, losts, learningRate, subTest = True, surfTest = True, placeTest=True, note=""):
    width = 0.35       # the width of the bars: can also be len(x) sequence    
    N = len(hists)    
    ## hists....    
    hists_keys=hists.keys()        
    losts_keys=losts.keys()
    
    T = len(hists[hists_keys[0]].ecs)

    sub_mu=np.empty(N)
    sub_sigma=np.empty(N)
    lost_mu=np.empty(N)
    lost_sigma=np.empty(N)
    
    hist_titles=[]
    #lost_titles=[]
    #gaussian posteriors for number of sense errors (np.std of belief in np.mean)
    hist_count=0    
    for hist_key in hists_keys:
        sub_mu[hist_count] = np.mean(hists[hist_key].sub_errs)
        sub_sigma[hist_count] =np.std(hists[hist_key].sub_errs)/np.sqrt(T)
        hist_titles.append(hists[hist_key].str_title)
        hist_count+=1

    #beta posteriors for lostness rates
    lost_count=0    
    for lost_key in losts_keys:
        (lost_mu[lost_count], lost_sigma[lost_count]) = cf_beta(T, np.sum(losts[lost_key]))
#        lost_titles.append(losts[lost_key].str_title)
        lost_count+=1

    if subTest:   #sense errors
        plt.figure()
        ind = np.arange(N)    # the x locations for the groups        
        p1 = plt.bar(ind, sub_mu,   width, color='w', yerr=sub_sigma)
        plt.ylabel('Sense errors'); 
        plt.xticks(ind+width/2., (hist_titles)); 
        plt.yticks(np.arange(0,1, .2))
        ti = "Denoised vs input sensor discrepencies\nDG: %s learningRate: %f" % (surfTest, learningRate )
        plt.title(ti)
        figName = "Results/err_senses_LR%d_DG%s_%s" % (learningRate*1000, surfTest, note)
       # COMMENTED AS RELATIVE PATHS HAVEN'T BEEN SORTED
       # plt.savefig(figName+".eps")
       # np.save(figName, np.vstack((np.array(sub_mu), np.array(sub_sigma))))

    if surfTest:
        #ALAN history files now have a ca1s attribute which are CA1State objects so hist1.ca1s[mapStep].surfs will give the cleaned up surf features
        #Compare hist.ca1s.surfs vs hist.surf_gnd_log.surfs
        #print("Step: %d\nCA1:\n%s\nSurf_gnd:\n%s" % (step, hist1.ca1s[step].surfs, hist1.surf_gnd_log[step]))
        hist_count=0  
        hist_accuracy = np.zeros(N)
        for hist_key in hists_keys:
            for step in range(len(hists[hist_key].ca1s)):
                hist_accuracy[hist_count] += DGStateAlan.accuracy(hists[hist_key].ca1s[step].surfs, hists[hist_key].surf_gnd_log[step])
            hist_accuracy[hist_count] = (hist_accuracy[hist_count] / float(len(hists[hist_key].ca1s)))*100
            hist_count+=1
            
        plt.figure()
        ind = np.arange(N)
        #accuracies = (hist1accuracy, hist2accuracy, hist3accuracy, hist4accuracy)
        plt.bar(ind,hist_accuracy, width, color='w')
        plt.ylabel('Accuracy EC vs CA1')
        plt.xticks(ind+width/2., hist_titles)
        plt.yticks(np.arange(0,100, 10))
        ti = "Original surf vs denoised surf\nDG: %s learningRate: %f" % (surfTest, learningRate )
        plt.title(ti)
        figName= "Results/err_surf_LR%d_DG%s_%s" % (learningRate*1000, surfTest, note)
        plt.savefig(figName + ".eps")
        np.save(figName, hist_accuracy)

    if placeTest:
        #NOW DO PLACE ERRORS
        plt.figure()
        #means  = (lost_mu_1, lost_mu_2, lost_mu_3, lost_mu_4)
        #sigmas = (lost_sigma_1, lost_sigma_2, lost_sigma_3, lost_sigma_4)
        ind = np.arange(N)    # the x locations for the groups
        p1 = plt.bar(ind, lost_mu,   width, color='w', yerr=lost_sigma)
        plt.ylabel('P(lost)')
        plt.xticks(ind+width/2., hist_titles)
        plt.yticks(np.arange(0,1.1, .1))
        ti = "Location error vs ground truth\nDG: %s learningRate: %f" % (surfTest, learningRate )
        plt.title(ti)
        figName = "Results/err_places_LR%d_DG%s_%s" % (learningRate*1000, surfTest, note)
        plt.savefig(figName + ".eps")
        np.save(figName, np.vstack((lost_mu, lost_sigma)))

def drawSquare(x,y):
    plt.plot( [x, x+1, x+1, x, x], [y, y, y+1, y+1, y],'k')

# Luke: As usual this is bloody fixed!!!!
def drawMaze():
    for x in range(0,7):
        drawSquare(x,3)
    for y in range(0,7):
        drawSquare(3,y)

def drawPath(xys_raw, color):
    e = 0.1*np.randn( xys_raw.shape[0], 2)  #add random deviations to prettify # Luke: this seems a bit wrong!?!
    xys = .5 + xys_raw+e   #.5 to center in boxes
    plt.plot(xys[:,0], xys[:,1], color)
    
