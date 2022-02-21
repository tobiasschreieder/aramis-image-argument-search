docker build --tag aramis-imarg:13 .
docker run -v /media:/media -v working:working aramis-imarg:13 -c -f -i /media/training-datasets/touche-task-3/touche-task-3-2022-01-21 -w working
docker run -v /media:/media -v /home/touche22-aramis/working:/working aramis-imarg:latest -f -i /media/training-datasets/touche-task-3/touche-task-3-2022-01-21 -w /working -qrel
docker run -v /media:/media -v /home/touche22-aramis/working:/working aramis-imarg:latest -f -i $inputDataset -o $outputDir -w /working -qrel -mtag "aramis|standard|standard|w0.5"
