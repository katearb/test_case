# Test case

В файле DialoguesParser.py находится имплементация класса DialoguesParser, с помощью которого решаются поставленные в задании задачи.
  * Метод **extract** принимает на вход изначальный датафрейм, возвращает также датафрейм, где находятся теги для каждой реплики: 
    * **greeting** и **goodbye** колонки содержат 1 в тех строках, где находят приветствия и прощания мееджера соотвественно. Эти данные извлекались из реплик с помощью регулярных выражений.
    * В колонке **manager_name** стоит извлеченное имя менеджера, которое он назвал при приветствии или 0, если имени в это реплике не найдено. Эта информация извлекалась с помощью правил Yargy.
    
     * В моем представлении, "менеджер представляет себя" и "менеджер называет свое имя" -- одна и та же информация. Поэтому колонка manager_self_represented имеет 1 в тех репликах, где удалось обнаружить имя менеджера.
    
    * В колонке **company_name** находится название компании. Эту информацию было сложнее всего извлечь правильно, поскольку на русском языке нет хороших доступных инстументов. Название компании извлекается с помощью правил Yargy, а также путем извлечения известных компаний из заранее заданного списка. Поэтому, чтобы этот метод работал хорошо, было бы замечательно иметь список тех компаний, которые могут встретиться в реплике. Однако некоторые названия способны извлекаться и без него.
   
   Результат работы этого метода лежит в файле result.csv
   
  * Метод get_report_for_dialogues представляет обобщенную информацию для каждого диалога: есть или нет приветствие и прощание, имя менеджера, название компании, а также в колонке check_passed стоит 1, если в диалоге менеджер и поздоровался, и попрощался, в любом другом случае -- 0.
    
    Этот метод возвращает датафрейм.
    
    Результат работы этого метода лежит в файле report_for_dialogues.csv
    
  Подробнее о классе и его методах рассказано в их строках документации.
    
В файле main.py находится реализация применения методов exctract и get_report_for_dialogues класса DialoguesParser.


### Перспективы улучшения пасера
1. При наличии большего количества данных, можно было бы обнаружить больше различных паттернов для лучшего покрытия различных кейсов.
2. Можно было бы улучшить парсер названий компаний, поскольку это довольно интересная задача, на которую требуется больше времени.
