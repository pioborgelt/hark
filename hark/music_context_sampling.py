"""Uniform train-origin selection and contiguous, duration-bounded music crops."""
import random


def contiguous_music_crop(backgrounds,minimum=100,maximum=400):
    if (minimum,maximum) not in [(100,400),(400,1199)]:
        raise ValueError('Registered short/long music windows only')
    noise=random.choice(backgrounds)
    if len(noise)<minimum:
        raise ValueError('Music excerpt shorter than the common minimum; never tile')
    length=random.randint(minimum,min(maximum,len(noise)))
    start=random.randrange(len(noise)-length+1)
    return noise[start:start+length].float()
