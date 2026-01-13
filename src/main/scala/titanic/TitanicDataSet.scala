package titanic

object TitanicDataSet {

  /**
   * Creates a model that predicts 1 (survived) if the person of the certain record
   * is female and 0 (deceased) otherwise
   *
   * @return The model represented as a function
   */
  def simpleModel: (Map[String, Any], String) => (Any, Any) =
    (record, idAttr) => {
      val id = record(idAttr) //Extrahiert die ID aus dem Datensatz
      val survived = if (record.getOrElse("sex", "") == "female") 1 else 0
      (id, survived)
    }
    //nimmt data input -> output tupe (id, prediction)
    //wenn geschlecht existiert und weiblich -> 1 sonst 0, also überleben oder nicht

  /**
   * This function should count for a given attribute list, how often an attribute is
   * not present in the data records of the data set
   *
   * @param data    The DataSet where the counting takes place
   * @param attList List of attributes where the missings should be counted
   * @return A Map that contains the attribute names (key) and the number of missings (value)
   */
  def countAllMissingValues(data: List[Map[String, Any]], attList: List[String]): Map[String, Int] =
    attList.map(a => (a, data.count(d => !d.contains(a)))).filter(x => x._2 > 0).toMap
    // !d ==... also wenn schlüssel a nicht in d exisitert
    // tupel (a, anzahl der fehlenden werte für dieses attribut)
    // filtert nur die attribute raus, die auch fehlende werte haben

  /**
   * This function should extract a set of given attributes from a record
   *
   * @param record  Record that should be extracted
   * @param attList List of attributes that should be extracted
   * @return A Map that contains only the attributes that should be extracted
   *
   */
  def extractTrainingAttributes(record: Map[String, Any], attList: List[String]): Map[String, Any] =
    record.filter(x => attList.contains(x._1))
    // ist gewünschtes attribut(der schlüssel) in der liste? true -> behalten, sonst wegwerfen

  /**
   * This function should create the training data set. It extracts the necessary attributes,
   * categorize them and deals with the missing values. You could find some hints in the description
   * and the lectures
   *
   * @param data Training Data Set that needs to be prepared
   * @return Prepared Data Set for using it with Naive Bayes
   */
  def createDataSetForTraining(data: List[Map[String, Any]]): List[Map[String, Any]] =
    val meanAge = data.flatMap(_.get("age")).map(_.toString.toDouble).sum / data.flatMap(_.get("age")).size
    data.map(record =>
      Map(
        "passengerID" -> record("passengerID"),
        "sex" -> record("sex"),
        "age" -> record.getOrElse("age", meanAge),
        "pclass" -> record("pclass"),
        "survived" -> record("survived")
      )
    )
    // transformiere jeden datensatz in neues format mit gewünschten attributen
    // wenn alter exisitert nutze es, sonst mean age

    // mittelwert für alter berechnen, alle age werte
    //mit .get sicherer zugriff
    // flatmap flacht die option werte ab
    // alter zu string zu double, any kann nicht direkt in double konvertiert werden
    // dann summe / anzahl der alters werte, wobei wieder flatmap zum abflachen von Option genutzt wird

  /**
   * This function builds the model. It is represented as a function that maps a data record
   * and the name of the id-attribute to the value of the id attribute and the predicted class
   * (similar to the model building process in the train example)
   *
   * @param trainDataSet Training Data Set
   * @param classAttrib  name of the attribute that contains the class
   * @return A tuple consisting of the id (first element) and the predicted class (second element)
   */
  def createModelWithTitanicTrainingData(tdata: List[Map[String, Any]], classAttr: String):
  (Map[String, Any], String) => (Any, Any) =
    NaiveBayes.modelwithAddOneSmoothing(createDataSetForTraining(tdata), classAttr)

    // classAttr = attribut dass wir vorhersagen wollen (survived)#
    // wie simpleModel aber mit machine learning,n utzt die naive bayes klasse
}