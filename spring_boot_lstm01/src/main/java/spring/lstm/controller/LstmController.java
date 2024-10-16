package spring.lstm.controller;

import java.nio.charset.Charset;

import org.apache.hc.client5.http.classic.methods.HttpPost;
import org.apache.hc.client5.http.impl.classic.CloseableHttpClient;
import org.apache.hc.client5.http.impl.classic.CloseableHttpResponse;
import org.apache.hc.client5.http.impl.classic.HttpClients;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.json.JSONArray;
import org.json.JSONObject;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.servlet.ModelAndView;

@Controller
public class LstmController {

	@RequestMapping(value="/webcamControl.do", method=RequestMethod.GET)
	public ModelAndView insertBookForm() {
		ModelAndView mav = new ModelAndView();
		mav.setViewName("webcam.html");
		return mav;
		
	}
	
	@ResponseBody
	
	@RequestMapping(value="/sendLstm.do", method=RequestMethod.POST)
	public String sendImageToRestServer(@RequestBody String data) throws Exception {
		data = data.replace("data:", "");
		System.out.println("data(after replace)=" + data);
		JSONObject dataObject = new JSONObject(data);
		System.out.println("dataObject=" + dataObject);
		JSONArray cam_data_arr = ((JSONArray)dataObject.get("img_data"));
		System.out.println("cam_data_arr=" + cam_data_arr);
		JSONObject restSendData = new JSONObject();
		restSendData.put("data", cam_data_arr);
		System.out.println("restSendData=" + restSendData);
		HttpPost httpPost = new HttpPost("http://localhost:5000/lstm_detect");
		httpPost.addHeader("Content-Type", "application/json;charset=utf-8");
		httpPost.setHeader("Accept", "application/json;charset=utf-8");
		StringEntity stringEntity = new StringEntity(restSendData.toString());
		httpPost.setEntity(stringEntity);
		CloseableHttpClient httpclient = HttpClients.createDefault();
		CloseableHttpResponse response2 = httpclient.execute(httpPost);
		String lstm_message = EntityUtils.toString(response2.getEntity(),Charset.forName("UTF-8"));
		return lstm_message;
	}
}
