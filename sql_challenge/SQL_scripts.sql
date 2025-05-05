--- Question 1: Day of the Week with the longest average trip duration ---
SELECT
DAYNAME(START_TIME) AS day_of_week,
AVG(DURATION_MINUTES) AS avg_trip_duration_minutes
FROM BIKERDATA
WHERE
    END_STATION_NAME NOT IN ('Missing', 'Stolen')
    AND START_STATION_ID IS NOT NULL
    AND END_STATION_ID IS NOT NULL
    AND START_STATION_ID != TRY_CAST(END_STATION_ID AS INTEGER)
GROUP BY 1
ORDER BY 2 DESC
LIMIT 1;


--- Question 2: Month-Year with most bike trips and the count ---
SELECT
    TO_CHAR(START_TIME, 'YYYY-MM') AS year_month,
    COUNT(*) AS trip_count
FROM 
    BIKERDATA
WHERE
    END_STATION_NAME NOT IN ('Missing', 'Stolen')
    AND START_STATION_ID IS NOT NULL
    AND END_STATION_ID IS NOT NULL
    AND START_STATION_ID != TRY_CAST(END_STATION_ID AS INTEGER)
GROUP BY 1
ORDER BY 2 DESC
LIMIT 1;
        
        
--- Question 3: Trips with the longest and shortest durations ---
WITH ranked_trips AS (
    SELECT
        *,
        ROW_NUMBER() OVER (ORDER BY DURATION_MINUTES DESC, START_TIME ASC) AS rt_longest,
        ROW_NUMBER() OVER (ORDER BY DURATION_MINUTES ASC, START_TIME ASC) AS rt_shortest
    FROM 
        BIKERDATA
    WHERE
        END_STATION_NAME NOT IN ('Missing', 'Stolen')
        AND START_STATION_ID IS NOT NULL
        AND END_STATION_ID IS NOT NULL
        AND START_STATION_ID != TRY_CAST(END_STATION_ID AS INTEGER)
)
SELECT 
    *
FROM 
    ranked_trips
WHERE
    rt_longest = 1 OR rt_shortest = 1;