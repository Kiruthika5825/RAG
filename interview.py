num=[123,1221,2244,2442]
str_num = str(num)

for i in num:
    if str(i) == str(i)[::-1]:
            print(f"{i} is a palindrome.")
    else:
            print(f"{i} is not a palindrome.")



string = "banana"
rev_string=""
for i in string:
       rev_string = i + rev_string

print(rev_string)

print("".join(reversed("apple")))

import string
statement = "Data is powerful. Data is science!"
statement=statement.lower()
for p in string.punctuation:
       statement=statement.replace(p,"")
list_words = statement.split()
word_count={}
for word in list_words:
       if word in word_count:
              word_count[word]+=1
       else:
              word_count[word]=1

print(word_count)


lst=[1,-10,5,20,-3,-20]
unique_lst= list(set(lst))
sorted_lst=sorted(unique_lst, reverse=True)
print(sorted_lst[1])

numbers = [1, 2, 2, 3, 4, 4, 5, 1]
unique = list(dict.fromkeys(numbers))
print(unique)

text = "interview"
text_count={}
for char in text:
       if char in text_count:
              text_count[char]+=1
       else:
             text_count[char]=1
print(text_count)


from collections import Counter
text_count = Counter(text)
print(text_count)

nested = [1, [2, 3], [4, [5, 6]], 7]
def flatten(nested):
       flat_list=[]
       for i in nested:
              if isinstance(i,list):
                     flat_list.extend(flatten(i))
              else:
                     flat_list.append(i)
       return flat_list

print(flatten(nested))


num=[1,2,3]
def add(*args):
      return sum(args)

print(add(*num))


dictionary = {"name": "kiruthika", "place": "tiruppur"}

def sample(**kwargs):
      for key, value in kwargs.items():
        print((key,value))
                                   

sample(**dictionary)


def dec(func):
       def wrapper():
              print("function started")
              func()
              print("function ended")
       return wrapper
@dec
def sample():
       print("This is a sample function.")

sample()

text = "DataScientist"
nums = [10, 20, 30, 40, 50]

print(text[4:])
print(nums[-3:])