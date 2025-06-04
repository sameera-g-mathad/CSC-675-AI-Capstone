import { Image, StyleSheet, FlatList, Text, Pressable, View } from "react-native";
import React, { useEffect, useState } from "react";
import { router } from 'expo-router'
import { ChatHistContextProvider } from "@/app/_context/ChathistContext";

const getChats = async () => {
    const response = await fetch('http://127.0.0.1:4000/api/v1/getChats', {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        },
    })
    const json = await response.json();
    return json.data;
}
export default function chat() {
    const [data, setData] = useState<{ chat_title: string, uuid: string, displayImage: string, _id: string }[]>([]);

    useEffect(() => {
        const getData = async () =>
            setData(await getChats())
        getData()
    }, [])
    return (
        <ChatHistContextProvider>
            <View style={styles.viewStyle}>
                <FlatList data={data} horizontal={false} keyExtractor={item => item.uuid} renderItem={({ item }) =>
                    <Pressable style={styles.pressableStyle} onPress={() => {
                        router.push({
                            pathname: `./chats/${item.uuid}`,
                            params: { chat_uuid: item.uuid }
                        });
                    }}>
                        <Image
                            source={{ uri: `https://ai-capstone-s3-bucket.s3.us-east-2.amazonaws.com/nst/${item.displayImage}` }}
                            style={styles.imageStyles}
                            resizeMode="cover" />
                        <Text style={styles.titleStyle}>{item.chat_title}</Text>
                    </Pressable>
                } />
            </View>
        </ChatHistContextProvider>
    );
}

const styles = StyleSheet.create({
    viewStyle: {
        flex: 1,
    },
    pressableStyle: {
        backgroundColor: 'white',
        padding: 20,
        margin: 5,
        borderRadius: '3%',
        display: 'flex',
        flexDirection: 'row',
        alignItems: 'center'
    },
    imageStyles: {
        width: 50,
        height: 50,
        borderRadius: '50%'
    },
    titleStyle: {
        fontSize: 17,
        paddingLeft: 10,
        letterSpacing: 0.5
    }
});
