using System.ComponentModel;
using Newtonsoft.Json;

namespace Codesigner.Models
{
    public class Output
    {
        [JsonProperty("id")]
        [Description("The id of the output parameter")]
        public string Id { get; set; }
        [JsonProperty("name")]
        [Description("The name of the output parameter.")]
        public string Name { get; set; }

        [JsonProperty("dataType")]
        [Description("The data type of the output (e.g., 'Number', 'Curve').")]
        public string DataType { get; set; }
    }
}