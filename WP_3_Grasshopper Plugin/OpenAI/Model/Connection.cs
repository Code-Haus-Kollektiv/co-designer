using System;
using System.Collections.Generic;
using System.ComponentModel;
using Newtonsoft.Json;


namespace Codesigner.Models
{
    public class Connection
    {
        [JsonProperty("fromComponent")]
        [Description("The id of the component where the connection originates.")]
        public string FromComponentId { get; set; }

        [JsonProperty("fromOutput")]
        [Description("The name of the output parameter on the source component.")]
        public string FromOutputName { get; set; }

        [JsonProperty("toComponent")]
        [Description("The id of the component receiving the connection.")]
        public string ToComponentId { get; set; }

        [JsonProperty("toInput")]
        [Description("The name of the input parameter on the destination component.")]
        public string ToInputName { get; set; }
    }

}


