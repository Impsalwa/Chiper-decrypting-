# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 17:51:34 2021

@author: Salwa
"""
import numpy as np 
import matplotlib as plt 
import string 
import random 
import re 
import requests
import os
import textwrap

###create substitution cipher 
#create two list one is the key other is the value
list1 = list(string.ascii_lowercase)
list2 = list(string.ascii_lowercase)
'''print(list1)
print(list2)'''

#dictionary contain the populate map
true_mapping = {}

#shuffle second set of  letters
random.shuffle(list2)

#populate map 
for k,v in zip(list1, list2):
    true_mapping[k] = v
#print(true_mapping)

### create the model language using markov model

#markov matrix the save all the bigrams probabilities 
M = np.ones((26, 26)) #26*26 propabilities
#vector for the unigram probabilities 
#initial state distribution
pi = np.zeros(26)

#function to update the markov matrix 
def update_transition(ch1, ch2):
    #ord its a function tha convert a carecter into integer 
    #ord 'a' = 97 'b' = 98 ... code ascii 
    i = ord(ch1)- 97 #mines the first letter 'a'
    j = ord(ch2)- 97
    M[i, j] += 1     
    
#function a update the initial state distribution 
def update_pi(ch):
    i = ord(ch) - 97
    pi[i]+= 1

#log probability of a word / token 
def get_word_prob(word):
    #to take the ungram probability by taking the first ch in a word 
    i = ord(word[0]) -97 #converting into int 
    logp = np.log(pi[i]) #took the pi prob and apply log function to get the log proba
    
    
    #loop for the rest of the word carecter 
    for ch in word[1:]: #start from the second charecter 
        j = ord(ch) - 97
        logp += np.log(M[i, j]) #update probability
        i = j #update the index word 
    return logp
# get the probability of a sequence of word 
def get_sequence_prob(words): #the input words could be a string contuning miltiole words or a list contuning miltiple words each store a string
    #if the input is string , split into list of tokens 
    if type(words) == str:
        words = words.split()
    logp = 0 
    for word in words:
        logp += get_word_prob(word)
    return logp 

### create a markov model based on an English datase

# download the file
'''if not os.path.exists('moby_dick.txt'):
  print("Downloading moby dick...")
  r = requests.get('https://lazyprogrammer.me/course_files/moby_dick.txt')
  with open('moby_dick.txt', 'w') as f:
    f.write(r.content.decode()'''
            
#populate the unigrams and bigrams data structures 
#remove any non alphabet characters
regex = re.compile('[^a-zA-Z]')

#looping the data line by line and delete the white space 
for line in open('moby_dick.txt',"r",encoding='utf-8'): #"r" and encoding for reading the file without errors we could use 'rb'to translates to read binary 
  line = line.rstrip()
  #in case of blank lines in the file 
  if line:
      line = regex.sub('', line) #replace all non alphabet charecters with space
      #split the tokens in the line and lowercase
      tokens = line.lower().split()
      for token in tokens:
          #update the model language 
          #first letter
          ch0 = token[0]
          update_pi(ch0)
          #for the rest start from the second letter 
          for ch1 in token[1:]:
              update_transition(ch0, ch1)
              ch0 = ch1
              
#normlize the probabilities
pi /= pi.sum()
M /= M.sum(axis=1, keepdims = True)
#pi is a vector of the unigram probabilities 
#print("pi",pi)
#M is matrix of bigrams probabilities
#print("M",M)

original_message = '''I then lounged down the street and found,
as I expected, that there was a mews in a lane which runs down
by one wall of the garden. I lent the ostlers a hand in rubbing
down their horses, and received in exchange twopence, a glass of
half-and-half, two fills of shag tobacco, and as much information
as I could desire about Miss Adler, to say nothing of half a dozen
other people in the neighbourhood in whom I was not in the least
interested, but whose biographies I was compelled to listen to.
'''
#function to encode a message
def encode_msg(msg):
    msg = msg.lower()
    #replace non alphabet characters 
    msg = regex.sub(' ', msg)
    
    #make the encode msg
    coded_msg = []
    for ch in msg:
        coded_ch = ch #could just be a space
        if ch in true_mapping:
            coded_ch = true_mapping[ch]
        coded_msg.append(coded_ch)
    return ''.join(coded_msg)
message ='I like dogs'
encoded_msg =  encode_msg(message)        
print("Encoding message is : ", encoded_msg)

#function to decode a message
def decode_msg(msg, word_map):
    decoded_msg = []
    for ch in msg:
        decoded_ch = ch 
        if ch in word_map:
            decoded_ch = word_map[ch]
        decoded_msg.append(decoded_ch)

    return ''.join(decoded_msg)

decoded_message = decode_msg(encoded_msg, true_mapping)
print("Decoding message is : ",decoded_message)


### the evolutionary algorithm to decode the message Genetic algorithm

# this is our initialization point
dna_pool = []
for _ in range(20):
  dna = list(string.ascii_lowercase)
  random.shuffle(dna)
  dna_pool.append(dna)
  
  
def evolve_offspring(dna_pool, n_children):
  # make n_children per offspring
  offspring = []

  for dna in dna_pool:
    for _ in range(n_children):
      copy = dna.copy()
      j = np.random.randint(len(copy))
      k = np.random.randint(len(copy))

      # sibstitution 
      tmp = copy[j]
      copy[j] = copy[k]
      copy[k] = tmp
      offspring.append(copy)
            
#the main loop that run Genetic algorithm
num_iters = 1000
scores = np.zeros(num_iters) #to store the avg scores of ech iteration of the loop 
best_dna = None
best_map = None
best_score = float('-inf') #- infinity witch is the minimum score 
for i in range(num_iters):
  if i > 0:
    # get offspring from the current dna pool
    dna_pool = evolve_offspring(dna_pool, 3)

  # calculate score for each dna
  dna2score = {}
  for dna in dna_pool:
    # populate map
    current_map = {}
    for k, v in zip(list1, dna):
      current_map[k] = v

    decoded_msg = decode_msg(encoded_msg, current_map)
    score = get_sequence_prob(decoded_msg)

    # store it
    # needs to be a string to be a dict key
    dna2score[''.join(dna)] = score

    # record the best so far
    if score > best_score:
      best_dna = dna
      best_map = current_map
      best_score = score

  # average score for this generation
  scores[i] = np.mean(list(dna2score.values()))

  # keep the best 5 dna
  # also turn them back into list of single chars
  sorted_dna = sorted(dna2score.items(), key=lambda x: x[1], reverse=True)
  dna_pool = [list(k) for k, v in sorted_dna[:5]]

  if i % 200 == 0:
    print("iter:", i, "score:", scores[i], "best so far:", best_score)

# use best score
decoded_message = decode_msg(encoded_msg, best_map)

print("LL of decoded message:", get_sequence_prob(decoded_message))
print("LL of true message:", get_sequence_prob(regex.sub(' ', original_message.lower())))


# which letters are wrong?
for true, v in true_mapping.items():
  pred = best_map[v]
  if true != pred:
    print("true: %s, pred: %s" % (true, pred))
    
# print the final decoded message
print("Decoded message:\n", textwrap.fill(decoded_message))

print("\nTrue message:\n", original_message)
            
            
#visulize the results
plt.plot(scores)
plt.show()
            
            
            
            
            
            
            
            
            
            
        