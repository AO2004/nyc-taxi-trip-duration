import pandas as pd
import numpy as np
import src.inference as infer





if __name__ == '__main__':

    while True:
        user_input = {
        'vendor_id' : [int(input('enter vendor id: '))],
        'pickup_datetime':[input('enter pickup date time: ')],
        'passenger_count' : [int(input('enter number of passengers: '))],
        'pickup_longitude' : [float(input('enter pickup longitude: '))],
        'pickup_latitude' : [float(input('enter pickup latitude: '))],
        'dropoff_longitude' : [float(input('enter dropoff longitude: '))],
        'dropoff_latitude' : [float(input('enter dropoff latitude: '))],
        'store_and_fwd_flag' : [input('enter store & forward: ')]
        }
        user_input = pd.DataFrame.from_dict(user_input)

        print(f'expected trip duration in seconds: {np.expm1(infer.inference(user_input))[0]: .2f} sec\n\n')

        is_continue = int(input('press key to continue, 0 to exit: '))
        if not is_continue:
            break