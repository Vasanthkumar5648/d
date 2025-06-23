import streamlit as st
st.title("🎯 Detection Summary Engine")
import cv2
import torch
import os
import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tempfile import NamedTemporaryFile
