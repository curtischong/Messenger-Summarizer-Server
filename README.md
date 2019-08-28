# Messenger-Summarizer-Server
Written by Flora Sun and Curtis Chong.

This is the server for Messenger Summarizer. Read what it can do here: [github.com/curtischong/Messenger-Summarizer](https://github.com/curtischong/Messenger-Summarizer).

# Installation (For Unix-based computers)

First you'll have to tell flask where the `main.py` file is.
So run `pwd` to get the current directory of the server.
For me, this is `/Users/curtis/Desktop/dev/Messenger-Summarizer-Server`.
Then I will open up my `.bash_profile` file (for mac) via `vim ~/.bash_profile`. If you are on linux, run `vim ~/.bashrc`.
Then export the FLASK_APP variable inside the file somewhere:

`export FLASK_APP=/Users/curtis/Desktop/dev/Messenger-Summarizer-Server/main.py`
(Don't forget to change my path `/Users/curtis/Desktop/dev/Messenger-Summarizer-Server` with the location of yours!)

Now when you leave vim, run `source ~/.bash_profile` (Mac) or `source ~/.bashrc` (Linux).
This will load the path of the server into your terminal session.

Now run `flask run` to turn on the server!