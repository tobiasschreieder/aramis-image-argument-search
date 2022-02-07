docker build --tag aramis-imarg:13 .
docker run -v /media:/media -v working:working aramis-imarg:13 -c -f -i /media/training-datasets/touche-task-3/touche-task-3-2022-01-21 -w working