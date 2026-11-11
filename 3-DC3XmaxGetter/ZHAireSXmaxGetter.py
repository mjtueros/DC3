import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DC3Common"))
import OutputDatabase
import utils

AiresInfo, AiresInp = utils.get_zhairestools_path()


def calculate_event_xmax(row, librarydir):
    """
    Calculate Xmax for one SQLite event.
    
    """
    ShowerName=row["ShowerName"]
    sryfile=librarydir+"/"+ShowerName+"/"+ShowerName+".sry"
    alt,distance,Xmaxx,Xmaxy,Xmaxz =AiresInfo.GetKmXmaxFromSry(sryfile)
    xmax=AiresInfo.GetSlantXmaxFromSry(sryfile) 
    return distance, alt, xmax
    
def calculate_event_em_energy_fraction(row, librarydir):    

    """
    Calculate Electromagnetic fraction of the event.
    
    This is an approximate value. We are adding up the energy deposited by all particles, 
    to the energy of all low energy particles of the simulation, that we consider end up depositing their energy

    this is an overestimation, as we are counting energy deposited by muons and hadrons, and low energy hadronic particles
    
    For inclined cascades, this is usually good enough. We are not counting particles getting to the ground.

    This can be done in much more detail, if needed. Sry file provides the info discriminated in gammas (stack1), e+/e- (stack2), heavy neutrals (stack3) and heavycharged (stack4)

    This would already provide an additional layer of detail, that would be more than enough. AiresInfo functions can be developed for that easily.
    
    """
    ShowerName=row["ShowerName"]
    sryfile=librarydir+"/"+ShowerName+"/"+ShowerName+".sry"
    depositedfraction=AiresInfo.GetEnergyFractionDepositedFromSry(sryfile)
    lostfraction=AiresInfo.GetEnergyFractionLostFromSry(sryfile)
    lowfraction=AiresInfo.GetEnergyFractionLowFromSry(sryfile)
    groundfraction=AiresInfo.GetEnergyFractionGroundFromSry(sryfile)
    neutrinofraction=AiresInfo.GetEnergyFractionInNeutrinosFromSry(sryfile) 
    
    totalinv=lostfraction+groundfraction+neutrinofraction
    totalem=depositedfraction+lowfraction
    total=totalinv+totalem

    #sanity checks
    if total<0.98 or total>1.02:
      print("total fraction is too high or too low",total)
    if neutrinofraction > 0.15:
      print("neutrino fraction seems too high", neutrinofraction)
    if groundfraction >0.15:   
      print("ground fraction seems too high, shower is underdeveloped?", groundfraction)
      
    #print("total",total,"totalem",totalem,"totalinv",totalinv)
    print("deposit",depositedfraction,"low",lowfraction,"lost",lostfraction,"ground",groundfraction,"neutrino",neutrinofraction)
    
    return totalem  


