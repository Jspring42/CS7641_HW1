from DT_cardio import main as DT_cardio
from DT_water import main as DT_water

from BT_cardio import main as BT_cardio
from BT_water import main as BT_water

from kNN_cardio import main as kNN_cardio
from kNN_water import main as kNN_water

from SVC_cardio import main as SVC_cardio
from SVC_water import main as SVC_water

from NN_cardio import main as NN_cardio
from NN_water import main as NN_water

def make_plots():

    print('Decision Trees')
    DT_cardio()
    DT_water()
    
    print('Boosted Trees')
    BT_cardio()
    BT_water()
    
    print('kNN')
    kNN_cardio()
    kNN_water()
    
    print('SVC')
    SVC_cardio()
    SVC_water()
    
    print('NN')
    NN_cardio()
    NN_water()

if __name__ == "__main__":
    
    make_plots()