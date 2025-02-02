using System;
using System.Collections.Generic;
using System.ComponentModel;
using Newtonsoft.Json;

namespace Codesigner.Models
{
    public class Component
    {
        [JsonProperty("id")]
        [Description("A unique identifier for the component.")]
        public string Id { get; set; }

        [JsonProperty("name")]
        [Description("The name of the component in Grasshopper (e.g., 'Rectangle', 'Number Slider').")]
        public string Name { get; set; }

        [JsonProperty("category")]
        [Description("Optional category or group in Grasshopper (e.g., 'Params', 'Math').")]
        public string Category { get; set; }

        [JsonProperty("position")]
        [Description("The position of the component on the Grasshopper canvas.")]
        public Position Position { get; set; }

        [JsonProperty("inputs")]
        [Description("List of input parameters for this component.")]
        public List<Input> Inputs { get; set; } = new List<Input>();

        [JsonProperty("outputs")]
        [Description("List of output parameters for this component.")]
        public List<Output> Outputs { get; set; } = new List<Output>();
    }


}