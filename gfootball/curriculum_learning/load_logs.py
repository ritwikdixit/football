import sys
import pickle

# timesteps
# ep reward mean
# ep len mean
# difficulty
def pretty_print(timesteps, eprewmean, eplenmean, difficulty):
  print('timesteps', timesteps)
  print('ep reward mean', eprewmean)
  print('ep len mean', eplenmean)
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
