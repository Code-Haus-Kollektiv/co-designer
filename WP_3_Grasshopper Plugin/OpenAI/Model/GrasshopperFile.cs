using System;
using System.Collections.Generic;
using System.ComponentModel;
using Newtonsoft.Json;

namespace Codesigner.Models
{
    public class GrasshopperSchema
    {
        [JsonProperty("components")]
        [Description("A list of all Grasshopper components on the canvas.")]
        public List<Component> Components { get; set; } = new List<Component>();

        [JsonProperty("connections")]
        [Description("A list of connections (wires) between component parameters.")]
        public List<Connection> Connections { get; set; } = new List<Connection>();
    }

}
