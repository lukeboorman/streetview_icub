import matplotlib
#matplotlib.use('Agg')
import numpy as np
from location import Location
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
    
    N_places = (1+2*path.N_mazeSize)**2

    for t in range(0,T):     
        loc=Location()
        loc.setGrids(hist.ca1s[t].grids, dictGrids)

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
    T=len(hist1.ecs)

    #gaussian posteriors for number of sense errors (np.std of belief in np.mean)
    sub_mu_1 = np.mean(hist1.sub_errs); sub_sigma_1 =np.std(hist1.sub_errs)/np.sqrt(T)
    sub_mu_2 = np.mean(hist2.sub_errs); sub_sigma_2 =np.std(hist2.sub_errs)/np.sqrt(T)
    sub_mu_3 = np.mean(hist3.sub_errs); sub_sigma_3 =np.std(hist3.sub_errs)/np.sqrt(T)

    #beta posteriors for lostness rates
    (lost_mu_1, lost_sigma_1) = cf_beta(T, np.sum(lost1))
    (lost_mu_2, lost_sigma_2) = cf_beta(T, np.sum(lost2))
    (lost_mu_3, lost_sigma_3) = cf_beta(T, np.sum(lost3))

    if subTest:   #sense errors
        plt.figure()

        N = 3
        means  = (sub_mu_1, sub_mu_2, sub_mu_3)
        sigmas = (sub_sigma_1, sub_sigma_2, sub_sigma_3)
        ind = np.arange(N)    # the x locations for the groups
        width = 0.35       # the width of the bars: can also be len(x) sequence
        p1 = plt.bar(ind, means,   width, color='w', yerr=sigmas)
        plt.ylabel('Sense errors'); 
        plt.xticks(ind+width/2., (hist1.str_title, hist2.str_title, hist3.str_title)); 
        plt.yticks(np.arange(0,1, .2))
        ti = "Denoised vs input sensor discrepencies\nDG: %s learningRate: %f" % (surfTest, learningRate )
        plt.title(ti)
        figName = "Results/err_senses_LR%d_DG%s_%s" % (learningRate*1000, surfTest, note)
       # COMMENTED AS RELATIVE PATHS HAVEN'T BEEN SORTED
       # plt.savefig(figName+".eps")
       # np.save(figName, np.vstack((np.array(means), np.array(sigmas))))

    if surfTest:
        #ALAN history files now have a ca1s attribute which are CA1State objects so hist1.ca1s[mapStep].surfs will give the cleaned up surf features
        #Compare hist.ca1s.surfs vs hist.surf_gnd_log.surfs
        #print("Step: %d\nCA1:\n%s\nSurf_gnd:\n%s" % (step, hist1.ca1s[step].surfs, hist1.surf_gnd_log[step]))
        hist1accuracy = 0
        for step in range(len(hist1.ca1s)):
            hist1accuracy += DGStateAlan.accuracy(hist1.ca1s[step].surfs, hist1.surf_gnd_log[step])
        hist1accuracy = (hist1accuracy / float(len(hist1.ca1s)))*100

        hist2accuracy = 0
        for step in range(len(hist2.ca1s)):
            hist2accuracy += DGStateAlan.accuracy(hist2.ca1s[step].surfs, hist2.surf_gnd_log[step])
        hist2accuracy = (hist2accuracy / float(len(hist2.ca1s)))*100

        hist3accuracy = 0
        for step in range(len(hist3.ca1s)):
            hist3accuracy += DGStateAlan.accuracy(hist3.ca1s[step].surfs, hist3.surf_gnd_log[step])
        hist3accuracy = (hist3accuracy / float(len(hist3.ca1s)))*100
        
        plt.figure()
        
        N=3
        ind = np.arange(N)
        accuracies = (hist1accuracy, hist2accuracy, hist3accuracy)
        width = 0.35
        plt.bar(ind,accuracies, width, color='w')
        plt.ylabel('Accuracy EC vs CA1')
        plt.xticks(ind+width/2., (hist1.str_title, hist2.str_title, hist3.str_title))
        plt.yticks(np.arange(0,100, 10))
        ti = "Original surf vs denoised surf\nDG: %s learningRate: %f" % (surfTest, learningRate )
        plt.title(ti)
        figName= "Results/err_surf_LR%d_DG%s_%s" % (learningRate*1000, surfTest, note)
        plt.savefig(figName + ".eps")
        np.save(figName, np.array(accuracies))

    if placeTest:
        #NOW DO PLACE ERRORS
        plt.figure()
        
        N = 3
        means  = (lost_mu_1, lost_mu_2, lost_mu_3)
        sigmas = (lost_sigma_1, lost_sigma_2, lost_sigma_3)
        ind = np.arange(N)    # the x locations for the groups
        width = 0.35       # the width of the bars: can also be len(x) sequence
        p1 = plt.bar(ind, means,   width, color='w', yerr=sigmas)
        plt.ylabel('P(lost)')
        plt.xticks(ind+width/2., (hist1.str_title, hist2.str_title, hist3.str_title))
        plt.yticks(np.arange(0,1.1, .1))
        ti = "Location error vs ground truth\nDG: %s learningRate: %f" % (surfTest, learningRate )
        plt.title(ti)
        figName = "Results/err_places_LR%d_DG%s_%s" % (learningRate*1000, surfTest, note)
        plt.savefig(figName + ".eps")
        np.save(figName, np.vstack((np.array(means), np.array(sigmas))))

def plotErrors4(hist1, hist2, hist3, hist4, lost1, lost2, lost3, lost4, learningRate, subTest = True, surfTest = True, placeTest=True, note=""):
    T=len(hist1.ecs)

    #gaussian posteriors for number of sense errors (np.std of belief in np.mean)
    sub_mu_1 = np.mean(hist1.sub_errs); sub_sigma_1 =np.std(hist1.sub_errs)/np.sqrt(T)
    sub_mu_2 = np.mean(hist2.sub_errs); sub_sigma_2 =np.std(hist2.sub_errs)/np.sqrt(T)
    sub_mu_3 = np.mean(hist3.sub_errs); sub_sigma_3 =np.std(hist3.sub_errs)/np.sqrt(T)
    sub_mu_4 = np.mean(hist4.sub_errs); sub_sigma_4 =np.std(hist4.sub_errs)/np.sqrt(T)

    #beta posteriors for lostness rates
    (lost_mu_1, lost_sigma_1) = cf_beta(T, np.sum(lost1))
    (lost_mu_2, lost_sigma_2) = cf_beta(T, np.sum(lost2))
    (lost_mu_3, lost_sigma_3) = cf_beta(T, np.sum(lost3))
    (lost_mu_4, lost_sigma_4) = cf_beta(T, np.sum(lost4))

    if subTest:   #sense errors
        plt.figure()

        N = 4
        means  = (sub_mu_1, sub_mu_2, sub_mu_3, sub_mu_4)
        sigmas = (sub_sigma_1, sub_sigma_2, sub_sigma_3, sub_sigma_4)
        ind = np.arange(N)    # the x locations for the groups
        width = 0.35       # the width of the bars: can also be len(x) sequence
        p1 = plt.bar(ind, means,   width, color='w', yerr=sigmas)
        plt.ylabel('Sense errors'); 
        plt.xticks(ind+width/2., (hist1.str_title, hist2.str_title, hist3.str_title, hist4.str_title)); 
        plt.yticks(np.arange(0,1, .2))
        ti = "Denoised vs input sensor discrepencies\nDG: %s learningRate: %f" % (surfTest, learningRate )
        plt.title(ti)
        figName = "Results/err_senses_LR%d_DG%s_%s" % (learningRate*1000, surfTest, note)
       # COMMENTED AS RELATIVE PATHS HAVEN'T BEEN SORTED
       # plt.savefig(figName+".eps")
       # np.save(figName, np.vstack((np.array(means), np.array(sigmas))))

    if surfTest:
        #ALAN history files now have a ca1s attribute which are CA1State objects so hist1.ca1s[mapStep].surfs will give the cleaned up surf features
        #Compare hist.ca1s.surfs vs hist.surf_gnd_log.surfs
        #print("Step: %d\nCA1:\n%s\nSurf_gnd:\n%s" % (step, hist1.ca1s[step].surfs, hist1.surf_gnd_log[step]))
        hist1accuracy = 0
        for step in range(len(hist1.ca1s)):
            hist1accuracy += DGStateAlan.accuracy(hist1.ca1s[step].surfs, hist1.surf_gnd_log[step])
        hist1accuracy = (hist1accuracy / float(len(hist1.ca1s)))*100

        hist2accuracy = 0
        for step in range(len(hist2.ca1s)):
            hist2accuracy += DGStateAlan.accuracy(hist2.ca1s[step].surfs, hist2.surf_gnd_log[step])
        hist2accuracy = (hist2accuracy / float(len(hist2.ca1s)))*100

        hist3accuracy = 0
        for step in range(len(hist3.ca1s)):
            hist3accuracy += DGStateAlan.accuracy(hist3.ca1s[step].surfs, hist3.surf_gnd_log[step])
        hist3accuracy = (hist3accuracy / float(len(hist3.ca1s)))*100

        hist4accuracy = 0
        for step in range(len(hist4.ca1s)):
            hist4accuracy += DGStateAlan.accuracy(hist4.ca1s[step].surfs, hist4.surf_gnd_log[step])
        hist4accuracy = (hist4accuracy / float(len(hist4.ca1s)))*100
        
        plt.figure()
        
        N=4
        ind = np.arange(N)
        accuracies = (hist1accuracy, hist2accuracy, hist3accuracy, hist4accuracy)
        width = 0.35
        plt.bar(ind,accuracies, width, color='w')
        plt.ylabel('Accuracy EC vs CA1')
        plt.xticks(ind+width/2., (hist1.str_title, hist2.str_title, hist3.str_title, hist4.str_title))
        plt.yticks(np.arange(0,100, 10))
        ti = "Original surf vs denoised surf\nDG: %s learningRate: %f" % (surfTest, learningRate )
        plt.title(ti)
        figName= "Results/err_surf_LR%d_DG%s_%s" % (learningRate*1000, surfTest, note)
        plt.savefig(figName + ".eps")
        np.save(figName, np.array(accuracies))

    if placeTest:
        #NOW DO PLACE ERRORS
        plt.figure()
        
        N = 4
        means  = (lost_mu_1, lost_mu_2, lost_mu_3, lost_mu_4)
        sigmas = (lost_sigma_1, lost_sigma_2, lost_sigma_3, lost_sigma_4)
        ind = np.arange(N)    # the x locations for the groups
        width = 0.35       # the width of the bars: can also be len(x) sequence
        p1 = plt.bar(ind, means,   width, color='w', yerr=sigmas)
        plt.ylabel('P(lost)')
        plt.xticks(ind+width/2., (hist1.str_title, hist2.str_title, hist3.str_title, hist4.str_title))
        plt.yticks(np.arange(0,1.1, .1))
        ti = "Location error vs ground truth\nDG: %s learningRate: %f" % (surfTest, learningRate )
        plt.title(ti)
        figName = "Results/err_places_LR%d_DG%s_%s" % (learningRate*1000, surfTest, note)
        plt.savefig(figName + ".eps")
        np.save(figName, np.vstack((np.array(means), np.array(sigmas))))

def drawSquare(x,y):
    plt.plot( [x, x+1, x+1, x, x], [y, y, y+1, y+1, y],'k')

def drawMaze():
    for x in range(0,7):
        drawSquare(x,3)
    for y in range(0,7):
        drawSquare(3,y)

def drawPath(xys_raw, color):
    e = 0.1*np.randn( xys_raw.shape[0], 2)  #add random deviations to prettify
    xys = .5 + xys_raw+e   #.5 to center in boxes
    plt.plot(xys[:,0], xys[:,1], color)
    
