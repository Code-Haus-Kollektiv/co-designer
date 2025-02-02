using System.ComponentModel;
using Newtonsoft.Json;

namespace Codesigner.Models
{
    public class Input
    {
        [JsonProperty("id")]
        [Description("The id of the input parameter")]
        public string Id { get; set; }
        [JsonProperty("name")]
        [Description("The name of the input parameter.")]
        public string Name { get; set; }

        [JsonProperty("dataType")]
        [Description("The expected data type (e.g., 'Number', 'Curve', 'Boolean').")]
        public string DataType { get; set; }

        [JsonProperty("defaultValue")]
        [Description("An optional default value for the input.")]
        public string DefaultValue { get; set; }
    }
}