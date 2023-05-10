from __future__ import unicode_literals
from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render
import json
from django.views.decorators.csrf import csrf_exempt
from user_model.models import user_registration as userreg

### This function is for fetching the user data.
def user_data(uname):
    try:
        user = userreg.objects.get(email=uname)
        return user
    except userreg.DoesNotExist:
        return 1
    
### This function is created for getting the username and password.
@csrf_exempt 
def user_info(request):
    if request.method == 'POST':
        email=request.POST.get("Email")
        respdata=user_data(email)
        if respdata is not 1:
            dict1 = {}
            dict1['First Name'] = respdata.fname
            dict1['Last Name'] = respdata.lname
            dict1['Mobile Number'] = respdata.mobile
            dict1['Email Id'] = respdata.email
            dict1['Address'] = respdata.address
            return HttpResponse(json.dumps(dict1), content_type = 'application/json')
        else:
            dict1 = {}
            dict1['message'] = 'Không tìm thấy'
            return HttpResponse(json.dumps(dict1), content_type = 'application/json')
    else:
        resp={}
        resp['status'] = 'Failed'
        resp['status_code'] = '400'
        resp['message'] = request.POST.get("Email")
        return HttpResponse(json.dumps(resp), content_type = 'application/json')
    # resp = {}
    # if request.method == 'POST':
    #     if 'application/json' in request.META['CONTENT_TYPE']:
    #         val1 = json.loads(request.body)
    #         uname = val1.get('Email')
    #         if uname:
    #             ## Calling the getting the user info.
    #             respdata = user_data(uname)
    #             dict1 = {}
    #             if respdata:
    #                 dict1['First Name'] = respdata.get('fname','')
    #                 dict1['Last Name'] = respdata.get('lname','')
    #                 dict1['Mobile Number'] = respdata.get('mobile','')
    #                 dict1['Email Id'] = respdata.get('email','')
    #                 dict1['Address'] = respdata.get('address','')
    #                 if dict1:
    #                     resp['status'] = 'Success'
    #                     resp['status_code'] = '200'
    #                     resp['data'] = dict1
    #                 ### If a user is not found then it give failed as a response.
    #                 else:
    #                     resp['status'] = 'Failed'
    #                     resp['status_code'] = '400'
    #                     resp['message'] = 'User Not Found.'
    #         ### The field value is missing.
    #             else:
    #                 resp['status'] = 'Failed'
    #                 resp['status_code'] = '400'
    #                 resp['message'] = 'Fields is mandatory.'
    #         else:
    #             resp['status'] = 'Failed'
    #             resp['status_code'] = '400'
    #             resp['message'] = 'Request type is not matched.'
    #     else:
    #         resp['status'] = 'Failed'
    #         resp['status_code'] = '400'
    #         resp['message'] = request.POST.get("Email")
    #         return HttpResponse(json.dumps(resp), content_type = 'application/json')
