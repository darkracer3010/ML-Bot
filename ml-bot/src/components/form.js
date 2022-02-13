
import React, { useState } from "react";
import axios from "axios";
import '../public/form.css'


const Form = ()=>{
    const [userName, setuserName] = useState('')
    const [noOfAttr, setnoOfAttr] = useState(0)
    const [typeofLearning, settypeofLearning] = useState('')
    const [dataLink, setdataLink] = useState('')
    const [djData,setdjData]=useState(0)
    const [testInput,settestInput]=useState()
    const [test,setTest]=useState(false)
    const data={
        name:userName,
        attr:noOfAttr,
        learning:typeofLearning,
        url:dataLink
    };
    const SubmitData=async ()=>{
        let infodata = new FormData()
        infodata.append("name",userName)
      infodata.append("attr", noOfAttr);
      infodata.append("learning", typeofLearning)
      infodata.append("url", dataLink);
      await axios({
          method: "post",
          url: "http://127.0.0.1:8000/api/predict",
        data: infodata
        }).then(res=>{
            console.log(res.data)
        });
        console.log(data)
    }
    const getData=async()=>{

        let testdata=new FormData()
        testdata.append("value",testInput)
        const resp = await axios({
          method: "get",
          url: "http://127.0.0.1:8000/api/getData",
          data:testdata,
          params:testInput
          
        })
      setdjData(resp.data.value);
      setTest(true)
      console.log(resp);
    }


    return (
      <div className="container box1 box2 background">
        <div className="box">
          <label>User Name</label>
          <input
            type="text"
            id="userName"
            name="username"
            placeholder="Enter your Name here"
            onChange={(e) => {
              setuserName(e.target.value);
            }}
          ></input>

          <label>No of Attributes Used</label>
          <input
            type="text"
            id="noattr"
            name="noattr"
            placeholder="No of Attr here.."
            onChange={(e) => {
              setnoOfAttr(e.target.value);
            }}
          ></input>

          <label>Type of Learning Required</label>
          <select
            id="learning"
            name="learning"
            onChange={(e) => {
              settypeofLearning(e.target.value);
            }}
          >
            <option selected value="Select One" disabled>
              Select One
            </option>
            <option value="Supervised">Supervised</option>
            <option value="Un-Supervised">Un-Supervised</option>
            <option value="Semi Supervised">Semi Supervised</option>
          </select>

          <label>Description About the Data</label>
          <textarea
            name="description"
            id="description"
            cols="25"
            rows="10"
            placeholder="Tell About the data used and purpose.."
          ></textarea>

          <label>Upload the File Here</label>
          <input
            type="text"
            id="data"
            name="data"
            onChange={(e) => {
              setdataLink(e.target.value);
            }}
          ></input>
          <br></br>
          <input type="submit" value="Send" onClick={SubmitData}></input>
          <input type="submit" value="Get results" onClick={getData}></input>
          <h3>{djData}</h3>
          <input
            type="text"
            onChange={(e) => {
              settestInput(e.target.value);
            }}
          ></input>
          
         
        </div>
      </div>
    );

}
export default Form
