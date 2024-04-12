sacct --format=JobID,JobName,Partition,State,Start,End,AllocNodes,AllocCPUs,NodeList --starttime 2024-01-01 --endtime 2024-04-20 --parsable2 > job_history.txt
