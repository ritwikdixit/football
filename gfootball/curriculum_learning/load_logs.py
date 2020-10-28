import sys
import pickle

# timesteps
# ep reward mean
# ep len mean
# difficulty
def pretty_print(timesteps, eprewmean_buf, rewlen, awsr, difficulty):
  print('timesteps', timesteps)
  print('ep reward mean (last 100 episodes)', eprewmean_buf)
  print('length of rewards arr', rewlen)
  print('sum of last (window_size=1e4,1e5) rewards', awsr)
  print('difficulty', difficulty)
  print('===')

if __name__ == '__main__':
    # use pickle path as first argument
    path = sys.argv[1]
    with open(path, 'rb') as pickle_file:
        logs_list = pickle.load(pickle_file)
        while True:
            try:
                print(pretty_print(*logs_list))
                logs_list = pickle.load(pickle_file)
            except EOFError:
                break
