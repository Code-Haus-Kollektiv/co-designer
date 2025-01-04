# Grasshopper Co-designer

Grasshopper Co-designer is a plugin for Rhino / Grasshopper that leverages a large language model (LLM) to assist computational designers in rapidly creating Grasshopper files. By predicting the next component in your workflow, it acts as a helpful assistant, streamlining the design process, enhancing productivity and teaching beginners.

---

## Features

- **Intelligent Predictions**: Suggests the next Grasshopper component based on your current workflow.
- **Seamless Integration**: Works directly within Grasshopper 6/7/8 to provide a smooth and intuitive user experience.
- **Private**: Everything runs locally. No data is sent to external servers.
- **Open-Source**: This repo contains our steps to create the model. You are always welcome to reach out to help us improve it!

---

## Installation

1. Download the latest release from the [Releases](https://github.com/your-repo/releases) page.
2. Extract the downloaded file.
3. Place the plugin files in your Grasshopper `Components` folder:
   - Windows: `C:\Users\<YourUsername>\AppData\Roaming\Grasshopper\Libraries`
   - macOS: `~/Library/Application Support/Grasshopper/Libraries`
4. Restart Rhino and Grasshopper.
5. The plugin should now appear in your Grasshopper interface.

---

## Usage

1. Open Grasshopper.
2. Start creating a new definition or open an existing one.
3. Add the *co-designer* component to your canvas.
4. As you add components, Co-designer will suggest the next logical component to use.
5. Accept suggestions with a single tab.

---

## Requirements

- Rhino 7 or higher.
- Grasshopper.

---

## Contributing

We welcome contributions to Grasshopper Copilot! To contribute:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`.
3. Commit your changes: `git commit -m 'Add some feature'`.
4. Push to the branch: `git push origin feature-name`.
5. Open a Pull Request.

Please ensure all contributions adhere to the [contribution guidelines](CONTRIBUTING.md).

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

Grasshopper co-designer was inspired by GitHub Copilot and aims to bring similar functionality to the design and architecture community using Grasshopper. Special thanks to the open-source community and everyone who contributed ideas and feedback during development.

---

## Contact

For support or questions, please open an issue on the [Issues](https://github.com/your-repo/issues) page or email us at support@grasshoppercopilot.com.

